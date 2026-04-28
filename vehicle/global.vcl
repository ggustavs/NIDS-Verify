--------------------------------------------------------------------------------
-- Inputs

-- We first define a new name for the type of inputs of the network.
-- In particular, it takes inputs of the form of a vector of m rational numbers
-- (the size depends).

m = 10

inputSize : Nat
inputSize = 2 + 4 * m

TCP = 0

outDirection = 0
inDirection = 1

FIN =  1 / 256
SYN =  2 / 256
RST =  4 / 256
PSH =  8 / 256
ACK = 16 / 256
URG = 32 / 256

size1    =  52 / 1000
size2    =  52 / 1000
size3    =  40 / 1000
minSize4 = 100 / 1000
maxSize4 = 500 / 1000
size5    =  40 / 1000

IATsThreshold = 0.05 -- seconds are: IATsThreshold * 50000000000
minTimeElapsed = 0.1 -- seconds are: IATsThreshold * 50000000000

type InputVector = Tensor Real [inputSize]

-- Next we add meaningful names for the indices.

timeElapsed       =  0
protocol          =  1

pktDirection1     =  2 -- 2 + 0 * m + 0
pktDirection2     =  3 -- 2 + 0 * m + 1
pktDirection3     =  4 -- 2 + 0 * m + 2
pktDirection4     =  5 -- 2 + 0 * m + 3
pktDirection5     =  6 -- 2 + 0 * m + 4
pktDirection6     =  7 -- 2 + 0 * m + 5
pktDirection7     =  8 -- 2 + 0 * m + 6
pktDirection8     =  9 -- 2 + 0 * m + 7
pktDirection9     = 10 -- 2 + 0 * m + 8
pktDirection10    = 11 -- 2 + 0 * m + 9

pktFlags1         = 12 -- 2 + 1 * m + 0
pktFlags2         = 13 -- 2 + 1 * m + 1
pktFlags3         = 14 -- 2 + 1 * m + 2
pktFlags4         = 15 -- 2 + 1 * m + 3
pktFlags5         = 16 -- 2 + 1 * m + 4
pktFlags6         = 17 -- 2 + 1 * m + 5
pktFlags7         = 18 -- 2 + 1 * m + 6
pktFlags8         = 19 -- 2 + 1 * m + 7
pktFlags9         = 20 -- 2 + 1 * m + 8
pktFlags10        = 21 -- 2 + 1 * m + 9

pktIATs1          = 22 -- 2 + 2 * m + 0
pktIATs2          = 23 -- 2 + 2 * m + 1
pktIATs3          = 24 -- 2 + 2 * m + 2
pktIATs4          = 25 -- 2 + 2 * m + 3
pktIATs5          = 26 -- 2 + 2 * m + 4
pktIATs6          = 27 -- 2 + 2 * m + 5
pktIATs7          = 28 -- 2 + 2 * m + 6
pktIATs8          = 29 -- 2 + 2 * m + 7
pktIATs9          = 30 -- 2 + 2 * m + 8
pktIATs10         = 31 -- 2 + 2 * m + 9

pktSize1          = 32 -- 2 + 3 * m + 0
pktSize2          = 33 -- 2 + 3 * m + 1
pktSize3          = 34 -- 2 + 3 * m + 2
pktSize4          = 35 -- 2 + 3 * m + 3
pktSize5          = 36 -- 2 + 3 * m + 4
pktSize6          = 37 -- 2 + 3 * m + 5
pktSize7          = 38 -- 2 + 3 * m + 6
pktSize8          = 39 -- 2 + 3 * m + 7
pktSize9          = 40 -- 2 + 3 * m + 8
pktSize10         = 41 -- 2 + 3 * m + 9

--------------------------------------------------------------------------------
-- Outputs

-- Outputs are a vector of 2 rationals. Representing the POS and NEG classes.

type OutputVector = Tensor Real [2]
type Label = Index 2

-- Again we define meaningful names for the indices into output vectors.

pos = 0
neg = 1

--------------------------------------------------------------------------------
-- The network

-- Next we use the `network` annotation to declare the name and the type of the
-- neural network we are verifying. The implementation is passed to the compiler
-- via a reference to the ONNX file at compile time.

@network
classifier : InputVector -> OutputVector

-- The classifier advises that input vector `x` has label `i` if the score
-- for label `i` is greater than the score of any other label `j`.
advises : InputVector -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

--------------------------------------------------------------------------------
-- Functions

-- Next we define the minimum and maximum values that each input can take.
-- These correspond to the range of the inputs that the network is designed
-- to work over.

checkPktFlags : Index inputSize -> Bool
checkPktFlags i = pktFlags1 <= i <= pktFlags10

checkIATs : Index inputSize -> Bool
checkIATs i = pktIATs2 <= i <= pktIATs10

checkSizes : Index inputSize -> Bool
checkSizes i = pktSize1 <= i <= pktSize10

-- We can therefore define a simple predicate saying whether a given input
-- vector is in the right range.

-- Valid Input

-- Instead of checking for every combination of packet directions, we can just consider the 5 most common combinations
-- Taking the 5 most common combinations (and assuming that the initial handshake compleetes correctly) we can write
-- a relaxed spefication that still captures approx. 70% of flows in training data whilst reducing our checks from 2048
-- to 8!
commonDirs : InputVector -> Bool
commonDirs x = x ! pktDirection1 == 0.0 and x ! pktDirection2 == 1.0 and (x ! pktDirection3 == 0.0) and x ! pktDirection4 == 0.0 and (x ! pktDirection5 == 1.0) and x ! pktDirection6 == 1.0 and x ! pktDirection7 == 1.0 and (x ! pktDirection8 == 1.0) and (x ! pktDirection9 == 1.0 or x ! pktDirection9 == 0.0) and (x ! pktDirection10 == 0.0) 

-- We can do something similar with flags
commonFlags : InputVector -> Bool
commonFlags x = x ! pktFlags1 == SYN and x ! pktFlags2 == SYN+ACK and x ! pktFlags3 == ACK and x ! pktFlags4 == PSH+ACK and x ! pktFlags5 == ACK and x ! pktFlags6 == ACK and x ! pktFlags7 == ACK and ( x ! pktFlags8 == ACK+PSH+FIN) and (x ! pktFlags9 == ACK+PSH+FIN) and (x ! pktFlags10 == ACK+PSH+FIN or x ! pktFlags10 == ACK+FIN)

validInput1 : InputVector -> Bool
validInput1 x = forall i . 0.0 <= x ! i <= 1.0

validInput2 : InputVector -> Bool 
validInput2 x = commonDirs x

validInput3 : InputVector -> Bool
validInput3 x = forall i . checkPktFlags i  => x ! i == 0.0 or x ! i == 0.015625 or x ! i == 0.03125 or x ! i == 0.046875 or x ! i == 0.0625 or x ! i == 0.078125 or x ! i == 0.09375 or x ! i == 0.109375 or x ! i == 0.125 or x ! i == 0.140625 or x ! i == 0.15625 or x ! i == 0.171875 or x ! i == 0.1875 or x ! i == 0.203125 or x ! i == 0.21875 or x ! i == 0.234375 or x ! i == 0.25 or x ! i == 0.265625 or x ! i == 0.28125 or x ! i == 0.296875 or x ! i == 0.3125 or x ! i == 0.328125 or x ! i == 0.34375 or x ! i == 0.359375 or x ! i == 0.375 or x ! i == 0.390625 or x ! i == 0.40625 or x ! i == 0.421875 or x ! i == 0.4375 or x ! i == 0.453125 or x ! i == 0.46875 or x ! i == 0.484375 or x ! i == 0.5 or x ! i == 0.515625 or x ! i == 0.53125 or x ! i == 0.546875 or x ! i == 0.5625 or x ! i == 0.578125 or x ! i == 0.59375 or x ! i == 0.609375 or x ! i == 0.625 or x ! i == 0.640625 or x ! i == 0.65625 or x ! i == 0.671875 or x ! i == 0.6875 or x ! i == 0.703125 or x ! i == 0.71875 or x ! i == 0.734375 or x ! i == 0.75 or x ! i == 0.765625 or x ! i == 0.78125 or x ! i == 0.796875 or x ! i == 0.8125 or x ! i == 0.828125 or x ! i == 0.84375 or x ! i == 0.859375 or x ! i == 0.875 or x ! i == 0.890625 or x ! i == 0.90625 or x ! i == 0.921875 or x ! i == 0.9375 or x ! i == 0.953125 or x ! i == 0.96875 or x ! i == 0.984375 or x ! i == 1.0

validInput4 : InputVector -> Bool
validInput4 x = x ! pktIATs1 == 0.0

-- validInput : InputVector -> Bool
-- validInput x = validInput1 x and validInput4 x

validInput : InputVector -> Bool
validInput x = validInput1 x and validInput2 x and commonFlags x and validInput4 x

-- validInput : InputVector -> Bool
-- validInput x = validInput1 x and validInput2 x and validInput3 x and validInput4 x

-- Valid TCP handshake

validTCPHandshake1 : InputVector -> Bool
validTCPHandshake1 x = x ! pktFlags1 == SYN and x ! pktSize1 == size1 and x ! pktDirection1 == outDirection

validTCPHandshake2 : InputVector -> Bool
validTCPHandshake2 x = x ! pktFlags2 == SYN + ACK and x ! pktSize2 == size2 and x ! pktDirection2 == inDirection

validTCPHandshake3 : InputVector -> Bool
validTCPHandshake3 x = x ! pktFlags3 == ACK and x ! pktSize3 == size3 and x ! pktDirection3 == outDirection

validTCPHandshake : InputVector -> Bool
validTCPHandshake x = validTCPHandshake1 x and validTCPHandshake2 x and validTCPHandshake3 x and x ! protocol == TCP

-- Invalid TCP handshake

INvalidTCPHandshake1 : InputVector -> Bool
INvalidTCPHandshake1 x = (SYN < x ! pktFlags1 < SYN) or (size1 < x ! pktSize1 < size1) or (outDirection < x ! pktDirection1 < outDirection)

INvalidTCPHandshake2 : InputVector -> Bool
INvalidTCPHandshake2 x = (SYN + ACK < x ! pktFlags2 < SYN + ACK) or (size2 < x ! pktSize2 < size2) or (inDirection < x ! pktDirection2 < inDirection)

INvalidTCPHandshake3 : InputVector -> Bool
INvalidTCPHandshake3 x = (ACK < x ! pktFlags3 < ACK) or (size3 < x ! pktSize3 < size3) and (outDirection < x ! pktDirection3 < outDirection)

INvalidTCPHandshake : InputVector -> Bool
INvalidTCPHandshake x = INvalidTCPHandshake1 x or INvalidTCPHandshake2 x or INvalidTCPHandshake3 x or (TCP < x ! protocol < TCP)

-- Valid HTTP get request

validHTTPGetRequest1 : InputVector -> Bool
validHTTPGetRequest1 x = x ! pktFlags4 == ACK + PSH and (minSize4 <= x ! pktSize4 <= maxSize4) and x ! pktDirection4 == outDirection

validHTTPGetRequest2 : InputVector -> Bool
validHTTPGetRequest2 x = x ! pktFlags5 == ACK and x ! pktSize5 == size5 and x ! pktDirection5 == inDirection

validHTTPGetRequest : InputVector -> Bool
validHTTPGetRequest x = validHTTPGetRequest1 x and validHTTPGetRequest2 x and x ! protocol == TCP

-- Invalid HTTP get request

INvalidHTTPGetRequest1 : InputVector -> Bool
INvalidHTTPGetRequest1 x = (ACK + PSH < x ! pktFlags4 < ACK + PSH) or (maxSize4 < x ! pktSize4 < minSize4) or (outDirection < x ! pktDirection4 < outDirection)

INvalidHTTPGetRequest2 : InputVector -> Bool
INvalidHTTPGetRequest2 x = (ACK < x ! pktFlags5 < ACK) or (size5 < x ! pktSize5 < size5) or (inDirection < x ! pktDirection5 < inDirection)

INvalidHTTPGetRequest : InputVector -> Bool
INvalidHTTPGetRequest x = INvalidHTTPGetRequest1 x or INvalidHTTPGetRequest2 x or (TCP < x ! protocol < TCP)

-- Valid Time Elapsed

validTimeElapsed : InputVector -> Bool
validTimeElapsed x = x ! timeElapsed == 0.0 or x ! timeElapsed >= minTimeElapsed

-- ValidIATs

validIATs : InputVector -> Bool
validIATs x = forall i . checkIATs i => (x ! i >= 0.01)

-- ValidSizes

validSizes : InputVector -> Bool
validSizes x = forall i . checkSizes i => (x ! i >= 40/1000 ) and (0.4 <= x ! pktSize10 + x ! pktSize9 + x ! pktSize8 + x ! pktSize7 + x ! pktSize6 + x ! pktSize5 )

-- Invalid Time Elapsed

INvalidTimeElapsed : InputVector -> Bool
INvalidTimeElapsed x = 0.000000000000000000001 <= x ! timeElapsed <= 0.000001

INvalidTimeElapsed2 : InputVector -> Bool
INvalidTimeElapsed2 x = 0.0000000000000000000000000000001 <= x ! timeElapsed <= 0.0001

-- Invalid IATs

INvalidIATs : InputVector -> Bool
INvalidIATs x = x ! pktIATs3 > IATsThreshold or x ! pktIATs4 > IATsThreshold or x ! pktIATs5 > IATsThreshold or x ! pktIATs9 > IATsThreshold or x ! pktIATs10 > IATsThreshold
-- INvalidIATs x = forall i . checkIATs i => (exists i . x ! i > 0.4)


-- BoundFlowDuration
-- We can sum the IATs and take them as being less than some threshold to bound flow BoundFlowDuration
boundFlowDuration : InputVector -> Bool
boundFlowDuration x = (forall i . checkIATs i => 0.0000001 <= x ! i <= 0.000005) and (0.00001 <= x ! pktIATs3 + x ! pktIATs6 + x ! pktIATs9  <= 0.000012)

-- BoundFlowDuration
-- We can sum the IATs and take them as being less than some threshold to bound flow BoundFlowDuration
notBoundFlowDuration : InputVector -> Bool
notBoundFlowDuration x = (forall i . checkIATs i => x ! i >= 0.1) and (x ! pktIATs3 + x ! pktIATs6 + x ! pktIATs9  >= 1.12)

--------------------------------------------------------------------------------
-----------------------------------GLOBAL PROPERTIES----------------------------
--------------------------------------------------------------------------------
-- Property Good

@property
propertyGoodHTTP : Bool
propertyGoodHTTP = forall x . validInput x and validTCPHandshake x and validHTTPGetRequest x and validTimeElapsed x and boundFlowDuration x and validSizes x =>
  advises x pos
--------------------------------------------------------------------------------
-- Property Invalid Inputs

@property
propertyInvalid : Bool
propertyInvalid = forall x . validInput x and (INvalidTCPHandshake x or INvalidHTTPGetRequest x) =>
  advises x neg
--------------------------------------------------------------------------------
-- Property Hulk attack

@property
propertyHulk : Bool
propertyHulk = forall x . validInput x and validTCPHandshake x and validIATs x and validSizes x and validHTTPGetRequest x and INvalidTimeElapsed x =>
 advises x neg
--------------------------------------------------------------------------------
-- Property SYN Flood attack

@property
propertySYNFlood : Bool
propertySYNFlood = forall x . validInput x and (INvalidTCPHandshake x or INvalidIATs x) =>
  advises x neg
--------------------------------------------------------------------------------
-- Property Slowhttptest attack

@property
propertySlowhttptest : Bool
propertySlowhttptest = forall x . validInput x and validTCPHandshake x and (INvalidHTTPGetRequest x or INvalidIATs x) =>
  advises x neg
--------------------------------------------------------------------------------
-- Property Slow IATs attacks

@property
propertySlowIATsAttacks : Bool
propertySlowIATsAttacks = forall x . validInput x and validTCPHandshake x and validSizes x and validHTTPGetRequest x and INvalidTimeElapsed2 x and notBoundFlowDuration x =>
  advises x neg
--------------------------------------------------------------------------------