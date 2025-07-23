-- NIDS Vehicle-lang Integration Configuration
-- Human-in-the-Loop Hyperrectangle Generation
-- Enhanced for formal verification of attack boundaries

--------------------------------------------------------------------------------
-- Network Input Configuration

-- Feature vector size based on NIDS preprocessing
m = 10  -- Number of packets in sequence
featureSize : Nat  
featureSize = 2 + 4 * m  -- Direction + timestamp + 4 features per packet

type FeatureVector = Vector Rat featureSize
type Label = Rat

-- Feature indices for documentation and constraints
directionIndex = 0
timestampIndex = 1

-- Per-packet feature indices (starting at index 2)
packetSizeIndex = 2
interArrivalTimeIndex = 3
flagsIndex = 4
protocolIndex = 5

--------------------------------------------------------------------------------
-- Network Protocols and Flags

TCP = 0
UDP = 1
ICMP = 2

-- Direction indicators
outDirection = 0
inDirection = 1

-- TCP Flags (normalized to [0,1])
FIN = 1 / 256
SYN = 2 / 256
RST = 4 / 256
PSH = 8 / 256
ACK = 16 / 256
URG = 32 / 256
ECE = 64 / 256
CWR = 128 / 256

--------------------------------------------------------------------------------
-- Attack Boundary Definitions
-- These can be generated using the Vehicle hyperrectangle generator

-- Basic validity constraints for input vectors
validInput : FeatureVector -> Bool
validInput x = 
  -- Direction must be 0 or 1
  (x ! directionIndex = 0 or x ! directionIndex = 1) and
  -- Timestamp must be positive
  x ! timestampIndex >= 0 and
  -- All packet features must be within reasonable bounds
  forall i . 
    (i >= 2 and i < featureSize) => 
      x ! i >= 0 and x ! i <= 1

-- DoS Attack Pattern (High volume, small packets)
dosAttackPattern : FeatureVector -> Bool
dosAttackPattern x =
  validInput x and
  -- High inter-arrival rate (small inter-arrival times)
  (forall i . 
    (i >= interArrivalTimeIndex and i < featureSize and (i - 2) % 4 = 1) =>
      x ! i <= 0.1) and
  -- Small packet sizes
  (forall i .
    (i >= packetSizeIndex and i < featureSize and (i - 2) % 4 = 0) =>
      x ! i <= 0.3)

-- DDoS Attack Pattern (Multiple sources, coordinated)
ddosAttackPattern : FeatureVector -> Bool  
ddosAttackPattern x =
  validInput x and
  -- Mixed directions (both incoming and outgoing)
  x ! directionIndex >= 0 and x ! directionIndex <= 1 and
  -- Moderate packet sizes
  (forall i .
    (i >= packetSizeIndex and i < featureSize and (i - 2) % 4 = 0) =>
      x ! i >= 0.2 and x ! i <= 0.8) and
  -- High volume characteristics
  (forall i .
    (i >= interArrivalTimeIndex and i < featureSize and (i - 2) % 4 = 1) =>
      x ! i <= 0.15)

-- Port Scan Attack Pattern (Sequential access patterns)
portScanPattern : FeatureVector -> Bool
portScanPattern x = 
  validInput x and
  -- Typically outgoing
  x ! directionIndex = outDirection and
  -- Small packets (probe packets)
  (forall i .
    (i >= packetSizeIndex and i < featureSize and (i - 2) % 4 = 0) =>
      x ! i <= 0.2) and
  -- SYN flags predominantly
  (forall i .
    (i >= flagsIndex and i < featureSize and (i - 2) % 4 = 2) =>
      x ! i >= SYN * 0.8)

-- Brute Force Attack Pattern (Repeated connection attempts)
bruteForcePattern : FeatureVector -> Bool
bruteForcePattern x =
  validInput x and
  -- Incoming direction
  x ! directionIndex = inDirection and
  -- High connection attempt rate
  (forall i .
    (i >= interArrivalTimeIndex and i < featureSize and (i - 2) % 4 = 1) =>
      x ! i <= 0.05) and
  -- Connection-oriented flags
  (forall i .
    (i >= flagsIndex and i < featureSize and (i - 2) % 4 = 2) =>
      x ! i >= SYN * 0.5)

--------------------------------------------------------------------------------
-- Combined Attack Regions

-- Union of all attack patterns
isInAttackRegion : FeatureVector -> Bool
isInAttackRegion x = 
  dosAttackPattern x or
  ddosAttackPattern x or  
  portScanPattern x or
  bruteForcePattern x

-- Benign traffic characteristics
benignTrafficPattern : FeatureVector -> Bool
benignTrafficPattern x =
  validInput x and
  not (isInAttackRegion x) and
  -- Normal inter-arrival times
  (forall i .
    (i >= interArrivalTimeIndex and i < featureSize and (i - 2) % 4 = 1) =>
      x ! i >= 0.2) and
  -- Normal packet sizes
  (forall i .
    (i >= packetSizeIndex and i < featureSize and (i - 2) % 4 = 0) =>
      x ! i >= 0.1 and x ! i <= 0.9)

--------------------------------------------------------------------------------
-- Network Declaration

-- NIDS classifier network (to be provided at verification time)
@network
nidsClassifier : FeatureVector -> Label

-- Helper function to interpret network output as attack classification
classifiesAsAttack : FeatureVector -> Bool
classifiesAsAttack x = nidsClassifier x >= 0.5

--------------------------------------------------------------------------------
-- Properties for Verification

-- Property 1: Attack Detection Completeness
-- If input matches known attack patterns, it should be classified as attack
@property
attackDetectionCompleteness : Bool
attackDetectionCompleteness = forall x .
  isInAttackRegion x => classifiesAsAttack x

-- Property 2: Benign Traffic Preservation  
-- Benign traffic should not be classified as attack (minimize false positives)
@property
benignTrafficPreservation : Bool
benignTrafficPreservation = forall x .
  benignTrafficPattern x => not (classifiesAsAttack x)

-- Property 3: DoS Specific Detection
-- DoS patterns specifically should be detected
@property
dosDetection : Bool
dosDetection = forall x .
  dosAttackPattern x => classifiesAsAttack x

-- Property 4: DDoS Specific Detection
-- DDoS patterns should be detected
@property
ddosDetection : Bool
ddosDetection = forall x .
  ddosAttackPattern x => classifiesAsAttack x

-- Property 5: Robustness - Small perturbations shouldn't change classification
-- This property checks adversarial robustness
perturbationRadius = 0.01

withinRadius : FeatureVector -> FeatureVector -> Bool
withinRadius x y = forall i .
  (x ! i - perturbationRadius <= y ! i) and (y ! i <= x ! i + perturbationRadius)

@property  
adversarialRobustness : Bool
adversarialRobustness = forall x . forall y .
  validInput x and validInput y and withinRadius x y =>
    (classifiesAsAttack x <=> classifiesAsAttack y)

--------------------------------------------------------------------------------
-- Utility Functions for Human-in-the-Loop Generation

-- Check if a point satisfies minimum attack characteristics
hasAttackCharacteristics : FeatureVector -> Bool
hasAttackCharacteristics x =
  -- High packet rate OR suspicious flags OR unusual patterns
  (exists i . 
    (i >= interArrivalTimeIndex and i < featureSize and (i - 2) % 4 = 1) and
    x ! i <= 0.1) or
  (exists i .
    (i >= flagsIndex and i < featureSize and (i - 2) % 4 = 2) and
    x ! i >= SYN) or
  (forall i .
    (i >= packetSizeIndex and i < featureSize and (i - 2) % 4 = 0) =>
    x ! i <= 0.2)

-- Feature importance scoring for human guidance
-- Higher scores indicate more discriminative features for attack detection
featureImportanceScore : FeatureVector -> Rat
featureImportanceScore x =
  -- Combine multiple factors that indicate attack likelihood
  let packetRateScore = if (exists i . 
                             (i >= interArrivalTimeIndex and i < featureSize and (i - 2) % 4 = 1) and
                             x ! i <= 0.1) then 0.3 else 0
      packetSizeScore = if (forall i .
                             (i >= packetSizeIndex and i < featureSize and (i - 2) % 4 = 0) =>
                             x ! i <= 0.3) then 0.25 else 0
      flagsScore = if (exists i .
                        (i >= flagsIndex and i < featureSize and (i - 2) % 4 = 2) and
                        x ! i >= SYN * 0.5) then 0.25 else 0
      directionScore = if x ! directionIndex = inDirection then 0.2 else 0.1
  in packetRateScore + packetSizeScore + flagsScore + directionScore

--------------------------------------------------------------------------------
-- Configuration for Vehicle-lang Tools

-- These parameters can be adjusted based on the specific dataset
-- and human expert knowledge

-- Confidence thresholds for automated suggestions
highConfidenceThreshold = 0.9
mediumConfidenceThreshold = 0.7
lowConfidenceThreshold = 0.5

-- Hyperrectangle size limits (prevent overly broad or narrow bounds)
minHyperrectangleVolume = 0.001
maxHyperrectangleVolume = 0.5

-- Feature scaling factors (for normalization)
maxPacketSize = 1500      -- bytes
maxInterArrivalTime = 10  -- seconds  
maxTimestamp = 3600       -- seconds

-- Attack type classifications
attackTypeDoS = 0
attackTypeDDoS = 1  
attackTypePortScan = 2
attackTypeBruteForce = 3
attackTypeOther = 4
