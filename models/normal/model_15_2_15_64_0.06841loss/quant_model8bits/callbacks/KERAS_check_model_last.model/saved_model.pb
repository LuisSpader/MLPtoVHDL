��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28ʏ
�
'quantize_layer_20/quantize_layer_20_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'quantize_layer_20/quantize_layer_20_min
�
;quantize_layer_20/quantize_layer_20_min/Read/ReadVariableOpReadVariableOp'quantize_layer_20/quantize_layer_20_min*
_output_shapes
: *
dtype0
�
'quantize_layer_20/quantize_layer_20_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'quantize_layer_20/quantize_layer_20_max
�
;quantize_layer_20/quantize_layer_20_max/Read/ReadVariableOpReadVariableOp'quantize_layer_20/quantize_layer_20_max*
_output_shapes
: *
dtype0
�
 quantize_layer_20/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quantize_layer_20/optimizer_step
�
4quantize_layer_20/optimizer_step/Read/ReadVariableOpReadVariableOp quantize_layer_20/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_68/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_68/optimizer_step
�
1quant_dense_68/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_68/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_68/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_68/kernel_min

-quant_dense_68/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_68/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_68/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_68/kernel_max

-quant_dense_68/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_68/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_68/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_68/post_activation_min
�
6quant_dense_68/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_68/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_68/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_68/post_activation_max
�
6quant_dense_68/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_68/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_69/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_69/optimizer_step
�
1quant_dense_69/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_69/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_69/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_69/kernel_min

-quant_dense_69/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_69/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_69/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_69/kernel_max

-quant_dense_69/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_69/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_69/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_69/post_activation_min
�
6quant_dense_69/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_69/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_69/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_69/post_activation_max
�
6quant_dense_69/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_69/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_70/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_70/optimizer_step
�
1quant_dense_70/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_70/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_70/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_70/kernel_min

-quant_dense_70/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_70/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_70/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_70/kernel_max

-quant_dense_70/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_70/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_70/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_70/post_activation_min
�
6quant_dense_70/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_70/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_70/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_70/post_activation_max
�
6quant_dense_70/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_70/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_71/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_71/optimizer_step
�
1quant_dense_71/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_71/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_71/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_71/kernel_min

-quant_dense_71/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_71/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_71/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_71/kernel_max

-quant_dense_71/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_71/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_71/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_71/post_activation_min
�
6quant_dense_71/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_71/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_71/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_71/post_activation_max
�
6quant_dense_71/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_71/post_activation_max*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
z
dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_68/kernel
s
#dense_68/kernel/Read/ReadVariableOpReadVariableOpdense_68/kernel*
_output_shapes

:@*
dtype0
r
dense_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_68/bias
k
!dense_68/bias/Read/ReadVariableOpReadVariableOpdense_68/bias*
_output_shapes
:*
dtype0
z
dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_69/kernel
s
#dense_69/kernel/Read/ReadVariableOpReadVariableOpdense_69/kernel*
_output_shapes

:*
dtype0
r
dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_69/bias
k
!dense_69/bias/Read/ReadVariableOpReadVariableOpdense_69/bias*
_output_shapes
:*
dtype0
z
dense_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_70/kernel
s
#dense_70/kernel/Read/ReadVariableOpReadVariableOpdense_70/kernel*
_output_shapes

:*
dtype0
r
dense_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_70/bias
k
!dense_70/bias/Read/ReadVariableOpReadVariableOpdense_70/bias*
_output_shapes
:*
dtype0
z
dense_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_71/kernel
s
#dense_71/kernel/Read/ReadVariableOpReadVariableOpdense_71/kernel*
_output_shapes

:@*
dtype0
r
dense_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_71/bias
k
!dense_71/bias/Read/ReadVariableOpReadVariableOpdense_71/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/dense_68/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_68/kernel/m
�
*Adam/dense_68/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/dense_68/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_68/bias/m
y
(Adam/dense_68/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_69/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_69/kernel/m
�
*Adam/dense_69/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_69/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_69/bias/m
y
(Adam/dense_69/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_70/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_70/kernel/m
�
*Adam/dense_70/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_70/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_70/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_70/bias/m
y
(Adam/dense_70/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_70/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_71/kernel/m
�
*Adam/dense_71/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/dense_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_71/bias/m
y
(Adam/dense_71/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_68/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_68/kernel/v
�
*Adam/dense_68/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/dense_68/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_68/bias/v
y
(Adam/dense_68/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_69/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_69/kernel/v
�
*Adam/dense_69/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_69/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_69/bias/v
y
(Adam/dense_69/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_69/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_70/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_70/kernel/v
�
*Adam/dense_70/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_70/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_70/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_70/bias/v
y
(Adam/dense_70/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_70/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_71/kernel/v
�
*Adam/dense_71/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/dense_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_71/bias/v
y
(Adam/dense_71/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_71/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
�X
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�W
value�WB�W B�W
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
 
�
quantize_layer_20_min
quantize_layer_20_max
quantizer_vars
optimizer_step
	variables
trainable_variables
regularization_losses
	keras_api
�
	layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers
	variables
trainable_variables
 regularization_losses
!	keras_api
�
	"layer
#optimizer_step
$_weight_vars
%
kernel_min
&
kernel_max
'_quantize_activations
(post_activation_min
)post_activation_max
*_output_quantizers
+	variables
,trainable_variables
-regularization_losses
.	keras_api
�
	/layer
0optimizer_step
1_weight_vars
2
kernel_min
3
kernel_max
4_quantize_activations
5post_activation_min
6post_activation_max
7_output_quantizers
8	variables
9trainable_variables
:regularization_losses
;	keras_api
�
	<layer
=optimizer_step
>_weight_vars
?
kernel_min
@
kernel_max
A_quantize_activations
Bpost_activation_min
Cpost_activation_max
D_output_quantizers
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
�
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_rateNm�Om�Pm�Qm�Rm�Sm�Tm�Um�Nv�Ov�Pv�Qv�Rv�Sv�Tv�Uv�
�
0
1
2
N3
O4
5
6
7
8
9
P10
Q11
#12
%13
&14
(15
)16
R17
S18
019
220
321
522
623
T24
U25
=26
?27
@28
B29
C30
8
N0
O1
P2
Q3
R4
S5
T6
U7
 
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
	trainable_variables

regularization_losses
 
��
VARIABLE_VALUE'quantize_layer_20/quantize_layer_20_minElayer_with_weights-0/quantize_layer_20_min/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'quantize_layer_20/quantize_layer_20_maxElayer_with_weights-0/quantize_layer_20_max/.ATTRIBUTES/VARIABLE_VALUE

min_var
max_var
tr
VARIABLE_VALUE quantize_layer_20/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
h

Nkernel
Obias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
qo
VARIABLE_VALUEquant_dense_68/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

d0
ig
VARIABLE_VALUEquant_dense_68/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_68/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_68/post_activation_minClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_68/post_activation_maxClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
1
N0
O1
2
3
4
5
6

N0
O1
 
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
 regularization_losses
h

Pkernel
Qbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
qo
VARIABLE_VALUEquant_dense_69/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

n0
ig
VARIABLE_VALUEquant_dense_69/kernel_min:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_69/kernel_max:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_69/post_activation_minClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_69/post_activation_maxClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
1
P0
Q1
#2
%3
&4
(5
)6

P0
Q1
 
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
+	variables
,trainable_variables
-regularization_losses
h

Rkernel
Sbias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
qo
VARIABLE_VALUEquant_dense_70/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

x0
ig
VARIABLE_VALUEquant_dense_70/kernel_min:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_70/kernel_max:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_70/post_activation_minClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_70/post_activation_maxClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
1
R0
S1
02
23
34
55
66

R0
S1
 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
8	variables
9trainable_variables
:regularization_losses
j

Tkernel
Ubias
~	variables
trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_71/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_71/kernel_min:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_71/kernel_max:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_71/post_activation_minClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_71/post_activation_maxClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
1
T0
U1
=2
?3
@4
B5
C6

T0
U1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_68/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_68/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_69/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_69/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_70/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_70/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_71/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_71/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
�
0
1
2
3
4
5
6
7
#8
%9
&10
(11
)12
013
214
315
516
617
=18
?19
@20
B21
C22
*
0
1
2
3
4
5

�0
 
 

0
1
2
 
 
 
 

O0

O0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses

N0
�2
#
0
1
2
3
4

0
 
 
 

Q0

Q0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses

P0
�2
#
#0
%1
&2
(3
)4

"0
 
 
 

S0

S0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses

R0
�2
#
00
21
32
53
64

/0
 
 
 

U0

U0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses

T0
�2
#
=0
?1
@2
B3
C4

<0
 
 
 
8

�total

�count
�	variables
�	keras_api
 
 
 
 
 

min_var
max_var
 
 
 
 
 

%min_var
&max_var
 
 
 
 
 

2min_var
3max_var
 
 
 
 
 

?min_var
@max_var
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
nl
VARIABLE_VALUEAdam/dense_68/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_68/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_69/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_69/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_70/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_70/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_71/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_71/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_68/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_68/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_69/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_69/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_70/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_70/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_71/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_71/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_21Placeholder*'
_output_shapes
:���������@*
dtype0*
shape:���������@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_21'quantize_layer_20/quantize_layer_20_min'quantize_layer_20/quantize_layer_20_maxdense_68/kernelquant_dense_68/kernel_minquant_dense_68/kernel_maxdense_68/bias"quant_dense_68/post_activation_min"quant_dense_68/post_activation_maxdense_69/kernelquant_dense_69/kernel_minquant_dense_69/kernel_maxdense_69/bias"quant_dense_69/post_activation_min"quant_dense_69/post_activation_maxdense_70/kernelquant_dense_70/kernel_minquant_dense_70/kernel_maxdense_70/bias"quant_dense_70/post_activation_min"quant_dense_70/post_activation_maxdense_71/kernelquant_dense_71/kernel_minquant_dense_71/kernel_maxdense_71/bias"quant_dense_71/post_activation_min"quant_dense_71/post_activation_max*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_4226969
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename;quantize_layer_20/quantize_layer_20_min/Read/ReadVariableOp;quantize_layer_20/quantize_layer_20_max/Read/ReadVariableOp4quantize_layer_20/optimizer_step/Read/ReadVariableOp1quant_dense_68/optimizer_step/Read/ReadVariableOp-quant_dense_68/kernel_min/Read/ReadVariableOp-quant_dense_68/kernel_max/Read/ReadVariableOp6quant_dense_68/post_activation_min/Read/ReadVariableOp6quant_dense_68/post_activation_max/Read/ReadVariableOp1quant_dense_69/optimizer_step/Read/ReadVariableOp-quant_dense_69/kernel_min/Read/ReadVariableOp-quant_dense_69/kernel_max/Read/ReadVariableOp6quant_dense_69/post_activation_min/Read/ReadVariableOp6quant_dense_69/post_activation_max/Read/ReadVariableOp1quant_dense_70/optimizer_step/Read/ReadVariableOp-quant_dense_70/kernel_min/Read/ReadVariableOp-quant_dense_70/kernel_max/Read/ReadVariableOp6quant_dense_70/post_activation_min/Read/ReadVariableOp6quant_dense_70/post_activation_max/Read/ReadVariableOp1quant_dense_71/optimizer_step/Read/ReadVariableOp-quant_dense_71/kernel_min/Read/ReadVariableOp-quant_dense_71/kernel_max/Read/ReadVariableOp6quant_dense_71/post_activation_min/Read/ReadVariableOp6quant_dense_71/post_activation_max/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#dense_68/kernel/Read/ReadVariableOp!dense_68/bias/Read/ReadVariableOp#dense_69/kernel/Read/ReadVariableOp!dense_69/bias/Read/ReadVariableOp#dense_70/kernel/Read/ReadVariableOp!dense_70/bias/Read/ReadVariableOp#dense_71/kernel/Read/ReadVariableOp!dense_71/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_68/kernel/m/Read/ReadVariableOp(Adam/dense_68/bias/m/Read/ReadVariableOp*Adam/dense_69/kernel/m/Read/ReadVariableOp(Adam/dense_69/bias/m/Read/ReadVariableOp*Adam/dense_70/kernel/m/Read/ReadVariableOp(Adam/dense_70/bias/m/Read/ReadVariableOp*Adam/dense_71/kernel/m/Read/ReadVariableOp(Adam/dense_71/bias/m/Read/ReadVariableOp*Adam/dense_68/kernel/v/Read/ReadVariableOp(Adam/dense_68/bias/v/Read/ReadVariableOp*Adam/dense_69/kernel/v/Read/ReadVariableOp(Adam/dense_69/bias/v/Read/ReadVariableOp*Adam/dense_70/kernel/v/Read/ReadVariableOp(Adam/dense_70/bias/v/Read/ReadVariableOp*Adam/dense_71/kernel/v/Read/ReadVariableOp(Adam/dense_71/bias/v/Read/ReadVariableOpConst*C
Tin<
:28	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_4228070
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename'quantize_layer_20/quantize_layer_20_min'quantize_layer_20/quantize_layer_20_max quantize_layer_20/optimizer_stepquant_dense_68/optimizer_stepquant_dense_68/kernel_minquant_dense_68/kernel_max"quant_dense_68/post_activation_min"quant_dense_68/post_activation_maxquant_dense_69/optimizer_stepquant_dense_69/kernel_minquant_dense_69/kernel_max"quant_dense_69/post_activation_min"quant_dense_69/post_activation_maxquant_dense_70/optimizer_stepquant_dense_70/kernel_minquant_dense_70/kernel_max"quant_dense_70/post_activation_min"quant_dense_70/post_activation_maxquant_dense_71/optimizer_stepquant_dense_71/kernel_minquant_dense_71/kernel_max"quant_dense_71/post_activation_min"quant_dense_71/post_activation_max	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_68/kerneldense_68/biasdense_69/kerneldense_69/biasdense_70/kerneldense_70/biasdense_71/kerneldense_71/biastotalcountAdam/dense_68/kernel/mAdam/dense_68/bias/mAdam/dense_69/kernel/mAdam/dense_69/bias/mAdam/dense_70/kernel/mAdam/dense_70/bias/mAdam/dense_71/kernel/mAdam/dense_71/bias/mAdam/dense_68/kernel/vAdam/dense_68/bias/vAdam/dense_69/kernel/vAdam/dense_69/bias/vAdam/dense_70/kernel/vAdam/dense_70/bias/vAdam/dense_71/kernel/vAdam/dense_71/bias/v*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_4228242��
�
�
0__inference_quant_dense_70_layer_call_fn_4227697

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4226311o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_quantize_layer_20_layer_call_fn_4227400

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4225942o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4226495

inputs=
+lastvaluequant_rank_readvariableop_resource:@/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0U
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :\
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0W
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :^
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ^
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�$
"__inference__wrapped_model_4225926
input_21f
\model_20_quantize_layer_20_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: h
^model_20_quantize_layer_20_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_20_quant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@b
Xmodel_20_quant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_20_quant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_20_quant_dense_68_biasadd_readvariableop_resource:c
Ymodel_20_quant_dense_68_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_20_quant_dense_68_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_20_quant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:b
Xmodel_20_quant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_20_quant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_20_quant_dense_69_biasadd_readvariableop_resource:c
Ymodel_20_quant_dense_69_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_20_quant_dense_69_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_20_quant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:b
Xmodel_20_quant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_20_quant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_20_quant_dense_70_biasadd_readvariableop_resource:c
Ymodel_20_quant_dense_70_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_20_quant_dense_70_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_20_quant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@b
Xmodel_20_quant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_20_quant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_20_quant_dense_71_biasadd_readvariableop_resource:@c
Ymodel_20_quant_dense_71_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_20_quant_dense_71_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��.model_20/quant_dense_68/BiasAdd/ReadVariableOp�Mmodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_20/quant_dense_69/BiasAdd/ReadVariableOp�Mmodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_20/quant_dense_70/BiasAdd/ReadVariableOp�Mmodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_20/quant_dense_71/BiasAdd/ReadVariableOp�Mmodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Smodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Umodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Smodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp\model_20_quantize_layer_20_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Umodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp^model_20_quantize_layer_20_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Dmodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinput_21[model_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0]model_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Mmodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_20_quant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
Omodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_20_quant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_20_quant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
model_20/quant_dense_68/MatMulMatMulNmodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
.model_20/quant_dense_68/BiasAdd/ReadVariableOpReadVariableOp7model_20_quant_dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_20/quant_dense_68/BiasAddBiasAdd(model_20/quant_dense_68/MatMul:product:06model_20/quant_dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_20/quant_dense_68/ReluRelu(model_20/quant_dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_20_quant_dense_68_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_20_quant_dense_68_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_20/quant_dense_68/Relu:activations:0Xmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Mmodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_20_quant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Omodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_20_quant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_20_quant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
model_20/quant_dense_69/MatMulMatMulKmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
.model_20/quant_dense_69/BiasAdd/ReadVariableOpReadVariableOp7model_20_quant_dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_20/quant_dense_69/BiasAddBiasAdd(model_20/quant_dense_69/MatMul:product:06model_20/quant_dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_20/quant_dense_69/ReluRelu(model_20/quant_dense_69/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_20_quant_dense_69_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_20_quant_dense_69_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_20/quant_dense_69/Relu:activations:0Xmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Mmodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_20_quant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Omodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_20_quant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_20_quant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
model_20/quant_dense_70/MatMulMatMulKmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
.model_20/quant_dense_70/BiasAdd/ReadVariableOpReadVariableOp7model_20_quant_dense_70_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_20/quant_dense_70/BiasAddBiasAdd(model_20/quant_dense_70/MatMul:product:06model_20/quant_dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_20/quant_dense_70/ReluRelu(model_20/quant_dense_70/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_20_quant_dense_70_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_20_quant_dense_70_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_20/quant_dense_70/Relu:activations:0Xmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Mmodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_20_quant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
Omodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_20_quant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_20_quant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
model_20/quant_dense_71/MatMulMatMulKmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
.model_20/quant_dense_71/BiasAdd/ReadVariableOpReadVariableOp7model_20_quant_dense_71_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_20/quant_dense_71/BiasAddBiasAdd(model_20/quant_dense_71/MatMul:product:06model_20/quant_dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Pmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_20_quant_dense_71_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_20_quant_dense_71_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars(model_20/quant_dense_71/BiasAdd:output:0Xmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityKmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp/^model_20/quant_dense_68/BiasAdd/ReadVariableOpN^model_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_20/quant_dense_69/BiasAdd/ReadVariableOpN^model_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_20/quant_dense_70/BiasAdd/ReadVariableOpN^model_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_20/quant_dense_71/BiasAdd/ReadVariableOpN^model_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1T^model_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpV^model_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.model_20/quant_dense_68/BiasAdd/ReadVariableOp.model_20/quant_dense_68/BiasAdd/ReadVariableOp2�
Mmodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_20/quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_20/quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_20/quant_dense_69/BiasAdd/ReadVariableOp.model_20/quant_dense_69/BiasAdd/ReadVariableOp2�
Mmodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_20/quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_20/quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_20/quant_dense_70/BiasAdd/ReadVariableOp.model_20/quant_dense_70/BiasAdd/ReadVariableOp2�
Mmodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_20/quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_20/quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_20/quant_dense_71/BiasAdd/ReadVariableOp.model_20/quant_dense_71/BiasAdd/ReadVariableOp2�
Mmodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_20/quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_20/quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Smodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpSmodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Umodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Umodel_20/quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_21
� 
�
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4226039

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�	
E__inference_model_20_layer_call_and_return_conditional_losses_4226088

inputs#
quantize_layer_20_4225943: #
quantize_layer_20_4225945: (
quant_dense_68_4225970:@ 
quant_dense_68_4225972:  
quant_dense_68_4225974: $
quant_dense_68_4225976: 
quant_dense_68_4225978:  
quant_dense_68_4225980: (
quant_dense_69_4226005: 
quant_dense_69_4226007:  
quant_dense_69_4226009: $
quant_dense_69_4226011: 
quant_dense_69_4226013:  
quant_dense_69_4226015: (
quant_dense_70_4226040: 
quant_dense_70_4226042:  
quant_dense_70_4226044: $
quant_dense_70_4226046: 
quant_dense_70_4226048:  
quant_dense_70_4226050: (
quant_dense_71_4226074:@ 
quant_dense_71_4226076:  
quant_dense_71_4226078: $
quant_dense_71_4226080:@ 
quant_dense_71_4226082:  
quant_dense_71_4226084: 
identity��&quant_dense_68/StatefulPartitionedCall�&quant_dense_69/StatefulPartitionedCall�&quant_dense_70/StatefulPartitionedCall�&quant_dense_71/StatefulPartitionedCall�)quantize_layer_20/StatefulPartitionedCall�
)quantize_layer_20/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_20_4225943quantize_layer_20_4225945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4225942�
&quant_dense_68/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_20/StatefulPartitionedCall:output:0quant_dense_68_4225970quant_dense_68_4225972quant_dense_68_4225974quant_dense_68_4225976quant_dense_68_4225978quant_dense_68_4225980*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4225969�
&quant_dense_69/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_68/StatefulPartitionedCall:output:0quant_dense_69_4226005quant_dense_69_4226007quant_dense_69_4226009quant_dense_69_4226011quant_dense_69_4226013quant_dense_69_4226015*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4226004�
&quant_dense_70/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_69/StatefulPartitionedCall:output:0quant_dense_70_4226040quant_dense_70_4226042quant_dense_70_4226044quant_dense_70_4226046quant_dense_70_4226048quant_dense_70_4226050*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4226039�
&quant_dense_71/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_70/StatefulPartitionedCall:output:0quant_dense_71_4226074quant_dense_71_4226076quant_dense_71_4226078quant_dense_71_4226080quant_dense_71_4226082quant_dense_71_4226084*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4226073~
IdentityIdentity/quant_dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_68/StatefulPartitionedCall'^quant_dense_69/StatefulPartitionedCall'^quant_dense_70/StatefulPartitionedCall'^quant_dense_71/StatefulPartitionedCall*^quantize_layer_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&quant_dense_68/StatefulPartitionedCall&quant_dense_68/StatefulPartitionedCall2P
&quant_dense_69/StatefulPartitionedCall&quant_dense_69/StatefulPartitionedCall2P
&quant_dense_70/StatefulPartitionedCall&quant_dense_70/StatefulPartitionedCall2P
&quant_dense_71/StatefulPartitionedCall&quant_dense_71/StatefulPartitionedCall2V
)quantize_layer_20/StatefulPartitionedCall)quantize_layer_20/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_4226969
input_21
unknown: 
	unknown_0: 
	unknown_1:@
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14: 

unknown_15: 

unknown_16:

unknown_17: 

unknown_18: 

unknown_19:@

unknown_20: 

unknown_21: 

unknown_22:@

unknown_23: 

unknown_24: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_4225926o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_21
�
�
*__inference_model_20_layer_call_fn_4227083

inputs
unknown: 
	unknown_0: 
	unknown_1:@
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14: 

unknown_15: 

unknown_16:

unknown_17: 

unknown_18: 

unknown_19:@

unknown_20: 

unknown_21: 

unknown_22:@

unknown_23: 

unknown_24: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_20_layer_call_and_return_conditional_losses_4226670o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�T
�
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4226219

inputs=
+lastvaluequant_rank_readvariableop_resource:@/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0U
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :\
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0W
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :^
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ^
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4225969

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4226073

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4226403

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0U
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :\
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0W
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :^
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ^
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_68_layer_call_fn_4227473

inputs
unknown:@
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4226495o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4227606

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�"
�
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4226543

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity��#AllValuesQuantize/AssignMaxAllValue�#AllValuesQuantize/AssignMinAllValue�8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�(AllValuesQuantize/Maximum/ReadVariableOp�(AllValuesQuantize/Minimum/ReadVariableOph
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: j
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0�
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�"
�
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4227439

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity��#AllValuesQuantize/AssignMaxAllValue�#AllValuesQuantize/AssignMinAllValue�8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�(AllValuesQuantize/Maximum/ReadVariableOp�(AllValuesQuantize/Minimum/ReadVariableOph
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: j
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0�
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_70_layer_call_fn_4227680

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4226039o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�-
E__inference_model_20_layer_call_and_return_conditional_losses_4227391

inputsM
Cquantize_layer_20_allvaluesquantize_minimum_readvariableop_resource: M
Cquantize_layer_20_allvaluesquantize_maximum_readvariableop_resource: L
:quant_dense_68_lastvaluequant_rank_readvariableop_resource:@>
4quant_dense_68_lastvaluequant_assignminlast_resource: >
4quant_dense_68_lastvaluequant_assignmaxlast_resource: <
.quant_dense_68_biasadd_readvariableop_resource:O
Equant_dense_68_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_68_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_69_lastvaluequant_rank_readvariableop_resource:>
4quant_dense_69_lastvaluequant_assignminlast_resource: >
4quant_dense_69_lastvaluequant_assignmaxlast_resource: <
.quant_dense_69_biasadd_readvariableop_resource:O
Equant_dense_69_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_69_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_70_lastvaluequant_rank_readvariableop_resource:>
4quant_dense_70_lastvaluequant_assignminlast_resource: >
4quant_dense_70_lastvaluequant_assignmaxlast_resource: <
.quant_dense_70_biasadd_readvariableop_resource:O
Equant_dense_70_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_70_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_71_lastvaluequant_rank_readvariableop_resource:@>
4quant_dense_71_lastvaluequant_assignminlast_resource: >
4quant_dense_71_lastvaluequant_assignmaxlast_resource: <
.quant_dense_71_biasadd_readvariableop_resource:@O
Equant_dense_71_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_71_movingavgquantize_assignmaxema_readvariableop_resource: 
identity��%quant_dense_68/BiasAdd/ReadVariableOp�+quant_dense_68/LastValueQuant/AssignMaxLast�+quant_dense_68/LastValueQuant/AssignMinLast�5quant_dense_68/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_68/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_68/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_68/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_68/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_68/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_69/BiasAdd/ReadVariableOp�+quant_dense_69/LastValueQuant/AssignMaxLast�+quant_dense_69/LastValueQuant/AssignMinLast�5quant_dense_69/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_69/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_69/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_69/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_69/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_69/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_70/BiasAdd/ReadVariableOp�+quant_dense_70/LastValueQuant/AssignMaxLast�+quant_dense_70/LastValueQuant/AssignMinLast�5quant_dense_70/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_70/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_70/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_70/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_70/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_70/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_71/BiasAdd/ReadVariableOp�+quant_dense_71/LastValueQuant/AssignMaxLast�+quant_dense_71/LastValueQuant/AssignMinLast�5quant_dense_71/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_71/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_71/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_71/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_71/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_71/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�5quantize_layer_20/AllValuesQuantize/AssignMaxAllValue�5quantize_layer_20/AllValuesQuantize/AssignMinAllValue�Jquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Lquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�:quantize_layer_20/AllValuesQuantize/Maximum/ReadVariableOp�:quantize_layer_20/AllValuesQuantize/Minimum/ReadVariableOpz
)quantize_layer_20/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
,quantize_layer_20/AllValuesQuantize/BatchMinMininputs2quantize_layer_20/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: |
+quantize_layer_20/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
,quantize_layer_20/AllValuesQuantize/BatchMaxMaxinputs4quantize_layer_20/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
:quantize_layer_20/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOpCquantize_layer_20_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
+quantize_layer_20/AllValuesQuantize/MinimumMinimumBquantize_layer_20/AllValuesQuantize/Minimum/ReadVariableOp:value:05quantize_layer_20/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: t
/quantize_layer_20/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-quantize_layer_20/AllValuesQuantize/Minimum_1Minimum/quantize_layer_20/AllValuesQuantize/Minimum:z:08quantize_layer_20/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
:quantize_layer_20/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOpCquantize_layer_20_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
+quantize_layer_20/AllValuesQuantize/MaximumMaximumBquantize_layer_20/AllValuesQuantize/Maximum/ReadVariableOp:value:05quantize_layer_20/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: t
/quantize_layer_20/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-quantize_layer_20/AllValuesQuantize/Maximum_1Maximum/quantize_layer_20/AllValuesQuantize/Maximum:z:08quantize_layer_20/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
5quantize_layer_20/AllValuesQuantize/AssignMinAllValueAssignVariableOpCquantize_layer_20_allvaluesquantize_minimum_readvariableop_resource1quantize_layer_20/AllValuesQuantize/Minimum_1:z:0;^quantize_layer_20/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0�
5quantize_layer_20/AllValuesQuantize/AssignMaxAllValueAssignVariableOpCquantize_layer_20_allvaluesquantize_maximum_readvariableop_resource1quantize_layer_20/AllValuesQuantize/Maximum_1:z:0;^quantize_layer_20/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0�
Jquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpCquantize_layer_20_allvaluesquantize_minimum_readvariableop_resource6^quantize_layer_20/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
Lquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCquantize_layer_20_allvaluesquantize_maximum_readvariableop_resource6^quantize_layer_20/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
;quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsRquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Tquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
1quant_dense_68/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_68_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0d
"quant_dense_68/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_68/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_68/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_68/LastValueQuant/rangeRange2quant_dense_68/LastValueQuant/range/start:output:0+quant_dense_68/LastValueQuant/Rank:output:02quant_dense_68/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_68/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_68_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
&quant_dense_68/LastValueQuant/BatchMinMin=quant_dense_68/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_68/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_68/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_68_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0f
$quant_dense_68/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_68/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_68/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_68/LastValueQuant/range_1Range4quant_dense_68/LastValueQuant/range_1/start:output:0-quant_dense_68/LastValueQuant/Rank_1:output:04quant_dense_68/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_68/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_68_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
&quant_dense_68/LastValueQuant/BatchMaxMax=quant_dense_68/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_68/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_68/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_68/LastValueQuant/truedivRealDiv/quant_dense_68/LastValueQuant/BatchMax:output:00quant_dense_68/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_68/LastValueQuant/MinimumMinimum/quant_dense_68/LastValueQuant/BatchMin:output:0)quant_dense_68/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_68/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_68/LastValueQuant/mulMul/quant_dense_68/LastValueQuant/BatchMin:output:0,quant_dense_68/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_68/LastValueQuant/MaximumMaximum/quant_dense_68/LastValueQuant/BatchMax:output:0%quant_dense_68/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_68/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_68_lastvaluequant_assignminlast_resource)quant_dense_68/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_68/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_68_lastvaluequant_assignmaxlast_resource)quant_dense_68/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_68_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_68_lastvaluequant_assignminlast_resource,^quant_dense_68/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_68_lastvaluequant_assignmaxlast_resource,^quant_dense_68/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
quant_dense_68/MatMulMatMulEquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_68/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_68/BiasAddBiasAddquant_dense_68/MatMul:product:0-quant_dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_68/ReluReluquant_dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:���������w
&quant_dense_68/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_68/MovingAvgQuantize/BatchMinMin!quant_dense_68/Relu:activations:0/quant_dense_68/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_68/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_68/MovingAvgQuantize/BatchMaxMax!quant_dense_68/Relu:activations:01quant_dense_68/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_68/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_68/MovingAvgQuantize/MinimumMinimum2quant_dense_68/MovingAvgQuantize/BatchMin:output:03quant_dense_68/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_68/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_68/MovingAvgQuantize/MaximumMaximum2quant_dense_68/MovingAvgQuantize/BatchMax:output:03quant_dense_68/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_68/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_68/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_68_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_68/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_68/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_68/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_68/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_68/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_68/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_68/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_68_movingavgquantize_assignminema_readvariableop_resource5quant_dense_68/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_68/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_68/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_68/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_68_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_68/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_68/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_68/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_68/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_68/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_68/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_68/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_68_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_68/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_68/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_68_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_68/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_68_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_68/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_68/Relu:activations:0Oquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
1quant_dense_69/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_69_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0d
"quant_dense_69/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_69/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_69/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_69/LastValueQuant/rangeRange2quant_dense_69/LastValueQuant/range/start:output:0+quant_dense_69/LastValueQuant/Rank:output:02quant_dense_69/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_69/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_69_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_69/LastValueQuant/BatchMinMin=quant_dense_69/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_69/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_69/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_69_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0f
$quant_dense_69/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_69/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_69/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_69/LastValueQuant/range_1Range4quant_dense_69/LastValueQuant/range_1/start:output:0-quant_dense_69/LastValueQuant/Rank_1:output:04quant_dense_69/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_69/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_69_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_69/LastValueQuant/BatchMaxMax=quant_dense_69/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_69/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_69/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_69/LastValueQuant/truedivRealDiv/quant_dense_69/LastValueQuant/BatchMax:output:00quant_dense_69/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_69/LastValueQuant/MinimumMinimum/quant_dense_69/LastValueQuant/BatchMin:output:0)quant_dense_69/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_69/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_69/LastValueQuant/mulMul/quant_dense_69/LastValueQuant/BatchMin:output:0,quant_dense_69/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_69/LastValueQuant/MaximumMaximum/quant_dense_69/LastValueQuant/BatchMax:output:0%quant_dense_69/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_69/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_69_lastvaluequant_assignminlast_resource)quant_dense_69/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_69/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_69_lastvaluequant_assignmaxlast_resource)quant_dense_69/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_69_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_69_lastvaluequant_assignminlast_resource,^quant_dense_69/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_69_lastvaluequant_assignmaxlast_resource,^quant_dense_69/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_69/MatMulMatMulBquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_69/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_69/BiasAddBiasAddquant_dense_69/MatMul:product:0-quant_dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_69/ReluReluquant_dense_69/BiasAdd:output:0*
T0*'
_output_shapes
:���������w
&quant_dense_69/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_69/MovingAvgQuantize/BatchMinMin!quant_dense_69/Relu:activations:0/quant_dense_69/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_69/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_69/MovingAvgQuantize/BatchMaxMax!quant_dense_69/Relu:activations:01quant_dense_69/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_69/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_69/MovingAvgQuantize/MinimumMinimum2quant_dense_69/MovingAvgQuantize/BatchMin:output:03quant_dense_69/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_69/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_69/MovingAvgQuantize/MaximumMaximum2quant_dense_69/MovingAvgQuantize/BatchMax:output:03quant_dense_69/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_69/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_69/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_69_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_69/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_69/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_69/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_69/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_69/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_69/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_69/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_69_movingavgquantize_assignminema_readvariableop_resource5quant_dense_69/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_69/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_69/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_69/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_69_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_69/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_69/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_69/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_69/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_69/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_69/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_69/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_69_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_69/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_69/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_69_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_69/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_69_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_69/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_69/Relu:activations:0Oquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
1quant_dense_70/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_70_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0d
"quant_dense_70/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_70/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_70/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_70/LastValueQuant/rangeRange2quant_dense_70/LastValueQuant/range/start:output:0+quant_dense_70/LastValueQuant/Rank:output:02quant_dense_70/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_70/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_70_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_70/LastValueQuant/BatchMinMin=quant_dense_70/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_70/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_70/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_70_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0f
$quant_dense_70/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_70/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_70/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_70/LastValueQuant/range_1Range4quant_dense_70/LastValueQuant/range_1/start:output:0-quant_dense_70/LastValueQuant/Rank_1:output:04quant_dense_70/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_70/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_70_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_70/LastValueQuant/BatchMaxMax=quant_dense_70/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_70/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_70/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_70/LastValueQuant/truedivRealDiv/quant_dense_70/LastValueQuant/BatchMax:output:00quant_dense_70/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_70/LastValueQuant/MinimumMinimum/quant_dense_70/LastValueQuant/BatchMin:output:0)quant_dense_70/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_70/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_70/LastValueQuant/mulMul/quant_dense_70/LastValueQuant/BatchMin:output:0,quant_dense_70/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_70/LastValueQuant/MaximumMaximum/quant_dense_70/LastValueQuant/BatchMax:output:0%quant_dense_70/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_70/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_70_lastvaluequant_assignminlast_resource)quant_dense_70/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_70/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_70_lastvaluequant_assignmaxlast_resource)quant_dense_70/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_70_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_70_lastvaluequant_assignminlast_resource,^quant_dense_70/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_70_lastvaluequant_assignmaxlast_resource,^quant_dense_70/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_70/MatMulMatMulBquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_70/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_70_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_70/BiasAddBiasAddquant_dense_70/MatMul:product:0-quant_dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_70/ReluReluquant_dense_70/BiasAdd:output:0*
T0*'
_output_shapes
:���������w
&quant_dense_70/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_70/MovingAvgQuantize/BatchMinMin!quant_dense_70/Relu:activations:0/quant_dense_70/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_70/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_70/MovingAvgQuantize/BatchMaxMax!quant_dense_70/Relu:activations:01quant_dense_70/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_70/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_70/MovingAvgQuantize/MinimumMinimum2quant_dense_70/MovingAvgQuantize/BatchMin:output:03quant_dense_70/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_70/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_70/MovingAvgQuantize/MaximumMaximum2quant_dense_70/MovingAvgQuantize/BatchMax:output:03quant_dense_70/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_70/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_70/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_70_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_70/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_70/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_70/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_70/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_70/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_70/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_70/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_70_movingavgquantize_assignminema_readvariableop_resource5quant_dense_70/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_70/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_70/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_70/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_70_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_70/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_70/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_70/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_70/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_70/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_70/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_70/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_70_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_70/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_70/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_70_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_70/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_70_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_70/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_70/Relu:activations:0Oquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
1quant_dense_71/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_71_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0d
"quant_dense_71/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_71/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_71/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_71/LastValueQuant/rangeRange2quant_dense_71/LastValueQuant/range/start:output:0+quant_dense_71/LastValueQuant/Rank:output:02quant_dense_71/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_71/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_71_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
&quant_dense_71/LastValueQuant/BatchMinMin=quant_dense_71/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_71/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_71/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_71_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0f
$quant_dense_71/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_71/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_71/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_71/LastValueQuant/range_1Range4quant_dense_71/LastValueQuant/range_1/start:output:0-quant_dense_71/LastValueQuant/Rank_1:output:04quant_dense_71/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_71/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_71_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
&quant_dense_71/LastValueQuant/BatchMaxMax=quant_dense_71/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_71/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_71/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_71/LastValueQuant/truedivRealDiv/quant_dense_71/LastValueQuant/BatchMax:output:00quant_dense_71/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_71/LastValueQuant/MinimumMinimum/quant_dense_71/LastValueQuant/BatchMin:output:0)quant_dense_71/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_71/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_71/LastValueQuant/mulMul/quant_dense_71/LastValueQuant/BatchMin:output:0,quant_dense_71/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_71/LastValueQuant/MaximumMaximum/quant_dense_71/LastValueQuant/BatchMax:output:0%quant_dense_71/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_71/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_71_lastvaluequant_assignminlast_resource)quant_dense_71/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_71/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_71_lastvaluequant_assignmaxlast_resource)quant_dense_71/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_71_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_71_lastvaluequant_assignminlast_resource,^quant_dense_71/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_71_lastvaluequant_assignmaxlast_resource,^quant_dense_71/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
quant_dense_71/MatMulMatMulBquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
%quant_dense_71/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_71_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_71/BiasAddBiasAddquant_dense_71/MatMul:product:0-quant_dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@w
&quant_dense_71/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_71/MovingAvgQuantize/BatchMinMinquant_dense_71/BiasAdd:output:0/quant_dense_71/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_71/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_71/MovingAvgQuantize/BatchMaxMaxquant_dense_71/BiasAdd:output:01quant_dense_71/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_71/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_71/MovingAvgQuantize/MinimumMinimum2quant_dense_71/MovingAvgQuantize/BatchMin:output:03quant_dense_71/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_71/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_71/MovingAvgQuantize/MaximumMaximum2quant_dense_71/MovingAvgQuantize/BatchMax:output:03quant_dense_71/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_71/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_71/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_71_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_71/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_71/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_71/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_71/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_71/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_71/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_71/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_71_movingavgquantize_assignminema_readvariableop_resource5quant_dense_71/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_71/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_71/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_71/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_71_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_71/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_71/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_71/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_71/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_71/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_71/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_71/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_71_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_71/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_71/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_71_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_71/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_71_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_71/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_71/BiasAdd:output:0Oquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityBquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp&^quant_dense_68/BiasAdd/ReadVariableOp,^quant_dense_68/LastValueQuant/AssignMaxLast,^quant_dense_68/LastValueQuant/AssignMinLast6^quant_dense_68/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_68/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_68/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_68/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_68/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_68/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_69/BiasAdd/ReadVariableOp,^quant_dense_69/LastValueQuant/AssignMaxLast,^quant_dense_69/LastValueQuant/AssignMinLast6^quant_dense_69/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_69/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_69/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_69/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_69/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_69/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_70/BiasAdd/ReadVariableOp,^quant_dense_70/LastValueQuant/AssignMaxLast,^quant_dense_70/LastValueQuant/AssignMinLast6^quant_dense_70/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_70/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_70/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_70/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_70/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_70/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_71/BiasAdd/ReadVariableOp,^quant_dense_71/LastValueQuant/AssignMaxLast,^quant_dense_71/LastValueQuant/AssignMinLast6^quant_dense_71/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_71/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_71/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_71/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_71/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_71/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_16^quantize_layer_20/AllValuesQuantize/AssignMaxAllValue6^quantize_layer_20/AllValuesQuantize/AssignMinAllValueK^quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpM^quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1;^quantize_layer_20/AllValuesQuantize/Maximum/ReadVariableOp;^quantize_layer_20/AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%quant_dense_68/BiasAdd/ReadVariableOp%quant_dense_68/BiasAdd/ReadVariableOp2Z
+quant_dense_68/LastValueQuant/AssignMaxLast+quant_dense_68/LastValueQuant/AssignMaxLast2Z
+quant_dense_68/LastValueQuant/AssignMinLast+quant_dense_68/LastValueQuant/AssignMinLast2n
5quant_dense_68/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_68/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_68/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_68/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_68/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_68/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_68/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_68/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_68/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_68/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_68/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_68/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_69/BiasAdd/ReadVariableOp%quant_dense_69/BiasAdd/ReadVariableOp2Z
+quant_dense_69/LastValueQuant/AssignMaxLast+quant_dense_69/LastValueQuant/AssignMaxLast2Z
+quant_dense_69/LastValueQuant/AssignMinLast+quant_dense_69/LastValueQuant/AssignMinLast2n
5quant_dense_69/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_69/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_69/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_69/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_69/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_69/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_69/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_69/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_69/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_69/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_69/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_69/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_70/BiasAdd/ReadVariableOp%quant_dense_70/BiasAdd/ReadVariableOp2Z
+quant_dense_70/LastValueQuant/AssignMaxLast+quant_dense_70/LastValueQuant/AssignMaxLast2Z
+quant_dense_70/LastValueQuant/AssignMinLast+quant_dense_70/LastValueQuant/AssignMinLast2n
5quant_dense_70/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_70/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_70/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_70/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_70/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_70/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_70/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_70/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_70/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_70/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_70/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_70/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_71/BiasAdd/ReadVariableOp%quant_dense_71/BiasAdd/ReadVariableOp2Z
+quant_dense_71/LastValueQuant/AssignMaxLast+quant_dense_71/LastValueQuant/AssignMaxLast2Z
+quant_dense_71/LastValueQuant/AssignMinLast+quant_dense_71/LastValueQuant/AssignMinLast2n
5quant_dense_71/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_71/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_71/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_71/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_71/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_71/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_71/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_71/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_71/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_71/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_71/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_71/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12n
5quantize_layer_20/AllValuesQuantize/AssignMaxAllValue5quantize_layer_20/AllValuesQuantize/AssignMaxAllValue2n
5quantize_layer_20/AllValuesQuantize/AssignMinAllValue5quantize_layer_20/AllValuesQuantize/AssignMinAllValue2�
Jquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Lquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Lquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12x
:quantize_layer_20/AllValuesQuantize/Maximum/ReadVariableOp:quantize_layer_20/AllValuesQuantize/Maximum/ReadVariableOp2x
:quantize_layer_20/AllValuesQuantize/Minimum/ReadVariableOp:quantize_layer_20/AllValuesQuantize/Minimum/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_71_layer_call_fn_4227792

inputs
unknown:@
	unknown_0: 
	unknown_1: 
	unknown_2:@
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4226073o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_model_20_layer_call_fn_4227026

inputs
unknown: 
	unknown_0: 
	unknown_1:@
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14: 

unknown_15: 

unknown_16:

unknown_17: 

unknown_18: 

unknown_19:@

unknown_20: 

unknown_21: 

unknown_22:@

unknown_23: 

unknown_24: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_20_layer_call_and_return_conditional_losses_4226088o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4227829

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�	
E__inference_model_20_layer_call_and_return_conditional_losses_4226904
input_21#
quantize_layer_20_4226846: #
quantize_layer_20_4226848: (
quant_dense_68_4226851:@ 
quant_dense_68_4226853:  
quant_dense_68_4226855: $
quant_dense_68_4226857: 
quant_dense_68_4226859:  
quant_dense_68_4226861: (
quant_dense_69_4226864: 
quant_dense_69_4226866:  
quant_dense_69_4226868: $
quant_dense_69_4226870: 
quant_dense_69_4226872:  
quant_dense_69_4226874: (
quant_dense_70_4226877: 
quant_dense_70_4226879:  
quant_dense_70_4226881: $
quant_dense_70_4226883: 
quant_dense_70_4226885:  
quant_dense_70_4226887: (
quant_dense_71_4226890:@ 
quant_dense_71_4226892:  
quant_dense_71_4226894: $
quant_dense_71_4226896:@ 
quant_dense_71_4226898:  
quant_dense_71_4226900: 
identity��&quant_dense_68/StatefulPartitionedCall�&quant_dense_69/StatefulPartitionedCall�&quant_dense_70/StatefulPartitionedCall�&quant_dense_71/StatefulPartitionedCall�)quantize_layer_20/StatefulPartitionedCall�
)quantize_layer_20/StatefulPartitionedCallStatefulPartitionedCallinput_21quantize_layer_20_4226846quantize_layer_20_4226848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4226543�
&quant_dense_68/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_20/StatefulPartitionedCall:output:0quant_dense_68_4226851quant_dense_68_4226853quant_dense_68_4226855quant_dense_68_4226857quant_dense_68_4226859quant_dense_68_4226861*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4226495�
&quant_dense_69/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_68/StatefulPartitionedCall:output:0quant_dense_69_4226864quant_dense_69_4226866quant_dense_69_4226868quant_dense_69_4226870quant_dense_69_4226872quant_dense_69_4226874*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4226403�
&quant_dense_70/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_69/StatefulPartitionedCall:output:0quant_dense_70_4226877quant_dense_70_4226879quant_dense_70_4226881quant_dense_70_4226883quant_dense_70_4226885quant_dense_70_4226887*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4226311�
&quant_dense_71/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_70/StatefulPartitionedCall:output:0quant_dense_71_4226890quant_dense_71_4226892quant_dense_71_4226894quant_dense_71_4226896quant_dense_71_4226898quant_dense_71_4226900*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4226219~
IdentityIdentity/quant_dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_68/StatefulPartitionedCall'^quant_dense_69/StatefulPartitionedCall'^quant_dense_70/StatefulPartitionedCall'^quant_dense_71/StatefulPartitionedCall*^quantize_layer_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&quant_dense_68/StatefulPartitionedCall&quant_dense_68/StatefulPartitionedCall2P
&quant_dense_69/StatefulPartitionedCall&quant_dense_69/StatefulPartitionedCall2P
&quant_dense_70/StatefulPartitionedCall&quant_dense_70/StatefulPartitionedCall2P
&quant_dense_71/StatefulPartitionedCall&quant_dense_71/StatefulPartitionedCall2V
)quantize_layer_20/StatefulPartitionedCall)quantize_layer_20/StatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_21
�
�
*__inference_model_20_layer_call_fn_4226782
input_21
unknown: 
	unknown_0: 
	unknown_1:@
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14: 

unknown_15: 

unknown_16:

unknown_17: 

unknown_18: 

unknown_19:@

unknown_20: 

unknown_21: 

unknown_22:@

unknown_23: 

unknown_24: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_20_layer_call_and_return_conditional_losses_4226670o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_21
�&
�	
E__inference_model_20_layer_call_and_return_conditional_losses_4226843
input_21#
quantize_layer_20_4226785: #
quantize_layer_20_4226787: (
quant_dense_68_4226790:@ 
quant_dense_68_4226792:  
quant_dense_68_4226794: $
quant_dense_68_4226796: 
quant_dense_68_4226798:  
quant_dense_68_4226800: (
quant_dense_69_4226803: 
quant_dense_69_4226805:  
quant_dense_69_4226807: $
quant_dense_69_4226809: 
quant_dense_69_4226811:  
quant_dense_69_4226813: (
quant_dense_70_4226816: 
quant_dense_70_4226818:  
quant_dense_70_4226820: $
quant_dense_70_4226822: 
quant_dense_70_4226824:  
quant_dense_70_4226826: (
quant_dense_71_4226829:@ 
quant_dense_71_4226831:  
quant_dense_71_4226833: $
quant_dense_71_4226835:@ 
quant_dense_71_4226837:  
quant_dense_71_4226839: 
identity��&quant_dense_68/StatefulPartitionedCall�&quant_dense_69/StatefulPartitionedCall�&quant_dense_70/StatefulPartitionedCall�&quant_dense_71/StatefulPartitionedCall�)quantize_layer_20/StatefulPartitionedCall�
)quantize_layer_20/StatefulPartitionedCallStatefulPartitionedCallinput_21quantize_layer_20_4226785quantize_layer_20_4226787*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4225942�
&quant_dense_68/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_20/StatefulPartitionedCall:output:0quant_dense_68_4226790quant_dense_68_4226792quant_dense_68_4226794quant_dense_68_4226796quant_dense_68_4226798quant_dense_68_4226800*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4225969�
&quant_dense_69/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_68/StatefulPartitionedCall:output:0quant_dense_69_4226803quant_dense_69_4226805quant_dense_69_4226807quant_dense_69_4226809quant_dense_69_4226811quant_dense_69_4226813*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4226004�
&quant_dense_70/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_69/StatefulPartitionedCall:output:0quant_dense_70_4226816quant_dense_70_4226818quant_dense_70_4226820quant_dense_70_4226822quant_dense_70_4226824quant_dense_70_4226826*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4226039�
&quant_dense_71/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_70/StatefulPartitionedCall:output:0quant_dense_71_4226829quant_dense_71_4226831quant_dense_71_4226833quant_dense_71_4226835quant_dense_71_4226837quant_dense_71_4226839*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4226073~
IdentityIdentity/quant_dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_68/StatefulPartitionedCall'^quant_dense_69/StatefulPartitionedCall'^quant_dense_70/StatefulPartitionedCall'^quant_dense_71/StatefulPartitionedCall*^quantize_layer_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&quant_dense_68/StatefulPartitionedCall&quant_dense_68/StatefulPartitionedCall2P
&quant_dense_69/StatefulPartitionedCall&quant_dense_69/StatefulPartitionedCall2P
&quant_dense_70/StatefulPartitionedCall&quant_dense_70/StatefulPartitionedCall2P
&quant_dense_71/StatefulPartitionedCall&quant_dense_71/StatefulPartitionedCall2V
)quantize_layer_20/StatefulPartitionedCall)quantize_layer_20/StatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_21
�
�
0__inference_quant_dense_71_layer_call_fn_4227809

inputs
unknown:@
	unknown_0: 
	unknown_1: 
	unknown_2:@
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4226219o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_quantize_layer_20_layer_call_fn_4227409

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4226543o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_model_20_layer_call_fn_4226143
input_21
unknown: 
	unknown_0: 
	unknown_1:@
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14: 

unknown_15: 

unknown_16:

unknown_17: 

unknown_18: 

unknown_19:@

unknown_20: 

unknown_21: 

unknown_22:@

unknown_23: 

unknown_24: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_20_layer_call_and_return_conditional_losses_4226088o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_21
�U
�
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4227551

inputs=
+lastvaluequant_rank_readvariableop_resource:@/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0U
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :\
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0W
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :^
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ^
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�h
�
 __inference__traced_save_4228070
file_prefixF
Bsavev2_quantize_layer_20_quantize_layer_20_min_read_readvariableopF
Bsavev2_quantize_layer_20_quantize_layer_20_max_read_readvariableop?
;savev2_quantize_layer_20_optimizer_step_read_readvariableop<
8savev2_quant_dense_68_optimizer_step_read_readvariableop8
4savev2_quant_dense_68_kernel_min_read_readvariableop8
4savev2_quant_dense_68_kernel_max_read_readvariableopA
=savev2_quant_dense_68_post_activation_min_read_readvariableopA
=savev2_quant_dense_68_post_activation_max_read_readvariableop<
8savev2_quant_dense_69_optimizer_step_read_readvariableop8
4savev2_quant_dense_69_kernel_min_read_readvariableop8
4savev2_quant_dense_69_kernel_max_read_readvariableopA
=savev2_quant_dense_69_post_activation_min_read_readvariableopA
=savev2_quant_dense_69_post_activation_max_read_readvariableop<
8savev2_quant_dense_70_optimizer_step_read_readvariableop8
4savev2_quant_dense_70_kernel_min_read_readvariableop8
4savev2_quant_dense_70_kernel_max_read_readvariableopA
=savev2_quant_dense_70_post_activation_min_read_readvariableopA
=savev2_quant_dense_70_post_activation_max_read_readvariableop<
8savev2_quant_dense_71_optimizer_step_read_readvariableop8
4savev2_quant_dense_71_kernel_min_read_readvariableop8
4savev2_quant_dense_71_kernel_max_read_readvariableopA
=savev2_quant_dense_71_post_activation_min_read_readvariableopA
=savev2_quant_dense_71_post_activation_max_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_dense_68_kernel_read_readvariableop,
(savev2_dense_68_bias_read_readvariableop.
*savev2_dense_69_kernel_read_readvariableop,
(savev2_dense_69_bias_read_readvariableop.
*savev2_dense_70_kernel_read_readvariableop,
(savev2_dense_70_bias_read_readvariableop.
*savev2_dense_71_kernel_read_readvariableop,
(savev2_dense_71_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_68_kernel_m_read_readvariableop3
/savev2_adam_dense_68_bias_m_read_readvariableop5
1savev2_adam_dense_69_kernel_m_read_readvariableop3
/savev2_adam_dense_69_bias_m_read_readvariableop5
1savev2_adam_dense_70_kernel_m_read_readvariableop3
/savev2_adam_dense_70_bias_m_read_readvariableop5
1savev2_adam_dense_71_kernel_m_read_readvariableop3
/savev2_adam_dense_71_bias_m_read_readvariableop5
1savev2_adam_dense_68_kernel_v_read_readvariableop3
/savev2_adam_dense_68_bias_v_read_readvariableop5
1savev2_adam_dense_69_kernel_v_read_readvariableop3
/savev2_adam_dense_69_bias_v_read_readvariableop5
1savev2_adam_dense_70_kernel_v_read_readvariableop3
/savev2_adam_dense_70_bias_v_read_readvariableop5
1savev2_adam_dense_71_kernel_v_read_readvariableop3
/savev2_adam_dense_71_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7BElayer_with_weights-0/quantize_layer_20_min/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/quantize_layer_20_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Bsavev2_quantize_layer_20_quantize_layer_20_min_read_readvariableopBsavev2_quantize_layer_20_quantize_layer_20_max_read_readvariableop;savev2_quantize_layer_20_optimizer_step_read_readvariableop8savev2_quant_dense_68_optimizer_step_read_readvariableop4savev2_quant_dense_68_kernel_min_read_readvariableop4savev2_quant_dense_68_kernel_max_read_readvariableop=savev2_quant_dense_68_post_activation_min_read_readvariableop=savev2_quant_dense_68_post_activation_max_read_readvariableop8savev2_quant_dense_69_optimizer_step_read_readvariableop4savev2_quant_dense_69_kernel_min_read_readvariableop4savev2_quant_dense_69_kernel_max_read_readvariableop=savev2_quant_dense_69_post_activation_min_read_readvariableop=savev2_quant_dense_69_post_activation_max_read_readvariableop8savev2_quant_dense_70_optimizer_step_read_readvariableop4savev2_quant_dense_70_kernel_min_read_readvariableop4savev2_quant_dense_70_kernel_max_read_readvariableop=savev2_quant_dense_70_post_activation_min_read_readvariableop=savev2_quant_dense_70_post_activation_max_read_readvariableop8savev2_quant_dense_71_optimizer_step_read_readvariableop4savev2_quant_dense_71_kernel_min_read_readvariableop4savev2_quant_dense_71_kernel_max_read_readvariableop=savev2_quant_dense_71_post_activation_min_read_readvariableop=savev2_quant_dense_71_post_activation_max_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_dense_68_kernel_read_readvariableop(savev2_dense_68_bias_read_readvariableop*savev2_dense_69_kernel_read_readvariableop(savev2_dense_69_bias_read_readvariableop*savev2_dense_70_kernel_read_readvariableop(savev2_dense_70_bias_read_readvariableop*savev2_dense_71_kernel_read_readvariableop(savev2_dense_71_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_68_kernel_m_read_readvariableop/savev2_adam_dense_68_bias_m_read_readvariableop1savev2_adam_dense_69_kernel_m_read_readvariableop/savev2_adam_dense_69_bias_m_read_readvariableop1savev2_adam_dense_70_kernel_m_read_readvariableop/savev2_adam_dense_70_bias_m_read_readvariableop1savev2_adam_dense_71_kernel_m_read_readvariableop/savev2_adam_dense_71_bias_m_read_readvariableop1savev2_adam_dense_68_kernel_v_read_readvariableop/savev2_adam_dense_68_bias_v_read_readvariableop1savev2_adam_dense_69_kernel_v_read_readvariableop/savev2_adam_dense_69_bias_v_read_readvariableop1savev2_adam_dense_70_kernel_v_read_readvariableop/savev2_adam_dense_70_bias_v_read_readvariableop1savev2_adam_dense_71_kernel_v_read_readvariableop/savev2_adam_dense_71_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : :@::::::@:@: : :@::::::@:@:@::::::@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:@: $

_output_shapes
:@:%

_output_shapes
: :&

_output_shapes
: :$' 

_output_shapes

:@: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
::$+ 

_output_shapes

:: ,

_output_shapes
::$- 

_output_shapes

:@: .

_output_shapes
:@:$/ 

_output_shapes

:@: 0

_output_shapes
::$1 

_output_shapes

:: 2

_output_shapes
::$3 

_output_shapes

:: 4

_output_shapes
::$5 

_output_shapes

:@: 6

_output_shapes
:@:7

_output_shapes
: 
� 
�
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4226004

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4226311

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0U
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :\
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0W
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :^
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ^
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_69_layer_call_fn_4227585

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4226403o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_68_layer_call_fn_4227456

inputs
unknown:@
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4225969o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4227775

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0U
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :\
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0W
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :^
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ^
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_69_layer_call_fn_4227568

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4226004o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4227663

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0U
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :\
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0W
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :^
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ^
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�	
E__inference_model_20_layer_call_and_return_conditional_losses_4226670

inputs#
quantize_layer_20_4226612: #
quantize_layer_20_4226614: (
quant_dense_68_4226617:@ 
quant_dense_68_4226619:  
quant_dense_68_4226621: $
quant_dense_68_4226623: 
quant_dense_68_4226625:  
quant_dense_68_4226627: (
quant_dense_69_4226630: 
quant_dense_69_4226632:  
quant_dense_69_4226634: $
quant_dense_69_4226636: 
quant_dense_69_4226638:  
quant_dense_69_4226640: (
quant_dense_70_4226643: 
quant_dense_70_4226645:  
quant_dense_70_4226647: $
quant_dense_70_4226649: 
quant_dense_70_4226651:  
quant_dense_70_4226653: (
quant_dense_71_4226656:@ 
quant_dense_71_4226658:  
quant_dense_71_4226660: $
quant_dense_71_4226662:@ 
quant_dense_71_4226664:  
quant_dense_71_4226666: 
identity��&quant_dense_68/StatefulPartitionedCall�&quant_dense_69/StatefulPartitionedCall�&quant_dense_70/StatefulPartitionedCall�&quant_dense_71/StatefulPartitionedCall�)quantize_layer_20/StatefulPartitionedCall�
)quantize_layer_20/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_20_4226612quantize_layer_20_4226614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4226543�
&quant_dense_68/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_20/StatefulPartitionedCall:output:0quant_dense_68_4226617quant_dense_68_4226619quant_dense_68_4226621quant_dense_68_4226623quant_dense_68_4226625quant_dense_68_4226627*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4226495�
&quant_dense_69/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_68/StatefulPartitionedCall:output:0quant_dense_69_4226630quant_dense_69_4226632quant_dense_69_4226634quant_dense_69_4226636quant_dense_69_4226638quant_dense_69_4226640*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4226403�
&quant_dense_70/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_69/StatefulPartitionedCall:output:0quant_dense_70_4226643quant_dense_70_4226645quant_dense_70_4226647quant_dense_70_4226649quant_dense_70_4226651quant_dense_70_4226653*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4226311�
&quant_dense_71/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_70/StatefulPartitionedCall:output:0quant_dense_71_4226656quant_dense_71_4226658quant_dense_71_4226660quant_dense_71_4226662quant_dense_71_4226664quant_dense_71_4226666*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4226219~
IdentityIdentity/quant_dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_68/StatefulPartitionedCall'^quant_dense_69/StatefulPartitionedCall'^quant_dense_70/StatefulPartitionedCall'^quant_dense_71/StatefulPartitionedCall*^quantize_layer_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&quant_dense_68/StatefulPartitionedCall&quant_dense_68/StatefulPartitionedCall2P
&quant_dense_69/StatefulPartitionedCall&quant_dense_69/StatefulPartitionedCall2P
&quant_dense_70/StatefulPartitionedCall&quant_dense_70/StatefulPartitionedCall2P
&quant_dense_71/StatefulPartitionedCall&quant_dense_71/StatefulPartitionedCall2V
)quantize_layer_20/StatefulPartitionedCall)quantize_layer_20/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4225942

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4227494

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
� 
E__inference_model_20_layer_call_and_return_conditional_losses_4227159

inputs]
Squantize_layer_20_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: _
Uquantize_layer_20_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@Y
Oquant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_68_biasadd_readvariableop_resource:Z
Pquant_dense_68_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_68_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:Y
Oquant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_69_biasadd_readvariableop_resource:Z
Pquant_dense_69_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_69_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:Y
Oquant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_70_biasadd_readvariableop_resource:Z
Pquant_dense_70_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_70_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@Y
Oquant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_71_biasadd_readvariableop_resource:@Z
Pquant_dense_71_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_71_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��%quant_dense_68/BiasAdd/ReadVariableOp�Dquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_69/BiasAdd/ReadVariableOp�Dquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_70/BiasAdd/ReadVariableOp�Dquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_71/BiasAdd/ReadVariableOp�Dquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Jquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Lquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Jquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpSquantize_layer_20_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Lquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpUquantize_layer_20_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
;quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsRquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Tquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Dquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_68_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
quant_dense_68/MatMulMatMulEquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_68/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_68/BiasAddBiasAddquant_dense_68/MatMul:product:0-quant_dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_68/ReluReluquant_dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Gquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_68_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_68_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_68/Relu:activations:0Oquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Dquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_69_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_69/MatMulMatMulBquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_69/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_69/BiasAddBiasAddquant_dense_69/MatMul:product:0-quant_dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_69/ReluReluquant_dense_69/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Gquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_69_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_69_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_69/Relu:activations:0Oquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Dquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_70_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_70/MatMulMatMulBquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_70/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_70_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_70/BiasAddBiasAddquant_dense_70/MatMul:product:0-quant_dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_70/ReluReluquant_dense_70/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Gquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_70_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_70_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_70/Relu:activations:0Oquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Dquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_71_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
quant_dense_71/MatMulMatMulBquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
%quant_dense_71/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_71_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_71/BiasAddBiasAddquant_dense_71/MatMul:product:0-quant_dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Gquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_71_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_71_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_71/BiasAdd:output:0Oquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityBquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp&^quant_dense_68/BiasAdd/ReadVariableOpE^quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_69/BiasAdd/ReadVariableOpE^quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_70/BiasAdd/ReadVariableOpE^quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_71/BiasAdd/ReadVariableOpE^quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1K^quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpM^quantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%quant_dense_68/BiasAdd/ReadVariableOp%quant_dense_68/BiasAdd/ReadVariableOp2�
Dquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_68/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_68/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_69/BiasAdd/ReadVariableOp%quant_dense_69/BiasAdd/ReadVariableOp2�
Dquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_69/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_69/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_70/BiasAdd/ReadVariableOp%quant_dense_70/BiasAdd/ReadVariableOp2�
Dquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_70/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_70/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_71/BiasAdd/ReadVariableOp%quant_dense_71/BiasAdd/ReadVariableOp2�
Dquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_71/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_71/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Jquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Lquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Lquantize_layer_20/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�T
�
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4227885

inputs=
+lastvaluequant_rank_readvariableop_resource:@/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0U
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :\
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0W
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :^
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ^
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4227718

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4227418

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�!
#__inference__traced_restore_4228242
file_prefixB
8assignvariableop_quantize_layer_20_quantize_layer_20_min: D
:assignvariableop_1_quantize_layer_20_quantize_layer_20_max: =
3assignvariableop_2_quantize_layer_20_optimizer_step: :
0assignvariableop_3_quant_dense_68_optimizer_step: 6
,assignvariableop_4_quant_dense_68_kernel_min: 6
,assignvariableop_5_quant_dense_68_kernel_max: ?
5assignvariableop_6_quant_dense_68_post_activation_min: ?
5assignvariableop_7_quant_dense_68_post_activation_max: :
0assignvariableop_8_quant_dense_69_optimizer_step: 6
,assignvariableop_9_quant_dense_69_kernel_min: 7
-assignvariableop_10_quant_dense_69_kernel_max: @
6assignvariableop_11_quant_dense_69_post_activation_min: @
6assignvariableop_12_quant_dense_69_post_activation_max: ;
1assignvariableop_13_quant_dense_70_optimizer_step: 7
-assignvariableop_14_quant_dense_70_kernel_min: 7
-assignvariableop_15_quant_dense_70_kernel_max: @
6assignvariableop_16_quant_dense_70_post_activation_min: @
6assignvariableop_17_quant_dense_70_post_activation_max: ;
1assignvariableop_18_quant_dense_71_optimizer_step: 7
-assignvariableop_19_quant_dense_71_kernel_min: 7
-assignvariableop_20_quant_dense_71_kernel_max: @
6assignvariableop_21_quant_dense_71_post_activation_min: @
6assignvariableop_22_quant_dense_71_post_activation_max: '
assignvariableop_23_adam_iter:	 )
assignvariableop_24_adam_beta_1: )
assignvariableop_25_adam_beta_2: (
assignvariableop_26_adam_decay: 0
&assignvariableop_27_adam_learning_rate: 5
#assignvariableop_28_dense_68_kernel:@/
!assignvariableop_29_dense_68_bias:5
#assignvariableop_30_dense_69_kernel:/
!assignvariableop_31_dense_69_bias:5
#assignvariableop_32_dense_70_kernel:/
!assignvariableop_33_dense_70_bias:5
#assignvariableop_34_dense_71_kernel:@/
!assignvariableop_35_dense_71_bias:@#
assignvariableop_36_total: #
assignvariableop_37_count: <
*assignvariableop_38_adam_dense_68_kernel_m:@6
(assignvariableop_39_adam_dense_68_bias_m:<
*assignvariableop_40_adam_dense_69_kernel_m:6
(assignvariableop_41_adam_dense_69_bias_m:<
*assignvariableop_42_adam_dense_70_kernel_m:6
(assignvariableop_43_adam_dense_70_bias_m:<
*assignvariableop_44_adam_dense_71_kernel_m:@6
(assignvariableop_45_adam_dense_71_bias_m:@<
*assignvariableop_46_adam_dense_68_kernel_v:@6
(assignvariableop_47_adam_dense_68_bias_v:<
*assignvariableop_48_adam_dense_69_kernel_v:6
(assignvariableop_49_adam_dense_69_bias_v:<
*assignvariableop_50_adam_dense_70_kernel_v:6
(assignvariableop_51_adam_dense_70_bias_v:<
*assignvariableop_52_adam_dense_71_kernel_v:@6
(assignvariableop_53_adam_dense_71_bias_v:@
identity_55��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7BElayer_with_weights-0/quantize_layer_20_min/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/quantize_layer_20_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp8assignvariableop_quantize_layer_20_quantize_layer_20_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp:assignvariableop_1_quantize_layer_20_quantize_layer_20_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp3assignvariableop_2_quantize_layer_20_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_quant_dense_68_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_quant_dense_68_kernel_minIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp,assignvariableop_5_quant_dense_68_kernel_maxIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp5assignvariableop_6_quant_dense_68_post_activation_minIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp5assignvariableop_7_quant_dense_68_post_activation_maxIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_quant_dense_69_optimizer_stepIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_quant_dense_69_kernel_minIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp-assignvariableop_10_quant_dense_69_kernel_maxIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp6assignvariableop_11_quant_dense_69_post_activation_minIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp6assignvariableop_12_quant_dense_69_post_activation_maxIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp1assignvariableop_13_quant_dense_70_optimizer_stepIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_quant_dense_70_kernel_minIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_quant_dense_70_kernel_maxIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_quant_dense_70_post_activation_minIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_quant_dense_70_post_activation_maxIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_quant_dense_71_optimizer_stepIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp-assignvariableop_19_quant_dense_71_kernel_minIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp-assignvariableop_20_quant_dense_71_kernel_maxIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_quant_dense_71_post_activation_minIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_quant_dense_71_post_activation_maxIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_iterIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_decayIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp&assignvariableop_27_adam_learning_rateIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_68_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_dense_68_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_69_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_dense_69_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_70_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp!assignvariableop_33_dense_70_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_71_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_71_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_totalIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_countIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_68_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_68_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_69_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_69_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_70_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_dense_70_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_71_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_71_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_68_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_68_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_69_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_69_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dense_70_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense_70_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_dense_71_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_dense_71_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_55Identity_55:output:0*�
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_211
serving_default_input_21:0���������@B
quant_dense_710
StatefulPartitionedCall:0���������@tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�
quantize_layer_20_min
quantize_layer_20_max
quantizer_vars
optimizer_step
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers
	variables
trainable_variables
 regularization_losses
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	"layer
#optimizer_step
$_weight_vars
%
kernel_min
&
kernel_max
'_quantize_activations
(post_activation_min
)post_activation_max
*_output_quantizers
+	variables
,trainable_variables
-regularization_losses
.	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	/layer
0optimizer_step
1_weight_vars
2
kernel_min
3
kernel_max
4_quantize_activations
5post_activation_min
6post_activation_max
7_output_quantizers
8	variables
9trainable_variables
:regularization_losses
;	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	<layer
=optimizer_step
>_weight_vars
?
kernel_min
@
kernel_max
A_quantize_activations
Bpost_activation_min
Cpost_activation_max
D_output_quantizers
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_rateNm�Om�Pm�Qm�Rm�Sm�Tm�Um�Nv�Ov�Pv�Qv�Rv�Sv�Tv�Uv�"
	optimizer
�
0
1
2
N3
O4
5
6
7
8
9
P10
Q11
#12
%13
&14
(15
)16
R17
S18
019
220
321
522
623
T24
U25
=26
?27
@28
B29
C30"
trackable_list_wrapper
X
N0
O1
P2
Q3
R4
S5
T6
U7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
	trainable_variables

regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
/:- 2'quantize_layer_20/quantize_layer_20_min
/:- 2'quantize_layer_20/quantize_layer_20_max
:
min_var
max_var"
trackable_dict_wrapper
(:& 2 quantize_layer_20/optimizer_step
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Nkernel
Obias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_68/optimizer_step
'
d0"
trackable_list_wrapper
!: 2quant_dense_68/kernel_min
!: 2quant_dense_68/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_68/post_activation_min
*:( 2"quant_dense_68/post_activation_max
 "
trackable_list_wrapper
Q
N0
O1
2
3
4
5
6"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
 regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Pkernel
Qbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_69/optimizer_step
'
n0"
trackable_list_wrapper
!: 2quant_dense_69/kernel_min
!: 2quant_dense_69/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_69/post_activation_min
*:( 2"quant_dense_69/post_activation_max
 "
trackable_list_wrapper
Q
P0
Q1
#2
%3
&4
(5
)6"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
+	variables
,trainable_variables
-regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Rkernel
Sbias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_70/optimizer_step
'
x0"
trackable_list_wrapper
!: 2quant_dense_70/kernel_min
!: 2quant_dense_70/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_70/post_activation_min
*:( 2"quant_dense_70/post_activation_max
 "
trackable_list_wrapper
Q
R0
S1
02
23
34
55
66"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
8	variables
9trainable_variables
:regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Tkernel
Ubias
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_71/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_71/kernel_min
!: 2quant_dense_71/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_71/post_activation_min
*:( 2"quant_dense_71/post_activation_max
 "
trackable_list_wrapper
Q
T0
U1
=2
?3
@4
B5
C6"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
!:@2dense_68/kernel
:2dense_68/bias
!:2dense_69/kernel
:2dense_69/bias
!:2dense_70/kernel
:2dense_70/bias
!:@2dense_71/kernel
:@2dense_71/bias
�
0
1
2
3
4
5
6
7
#8
%9
&10
(11
)12
013
214
315
516
617
=18
?19
@20
B21
C22"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
O0"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
N0
�2"
trackable_tuple_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
Q0"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
P0
�2"
trackable_tuple_wrapper
C
#0
%1
&2
(3
)4"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
S0"
trackable_list_wrapper
'
S0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
R0
�2"
trackable_tuple_wrapper
C
00
21
32
53
64"
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
U0"
trackable_list_wrapper
'
U0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
T0
�2"
trackable_tuple_wrapper
C
=0
?1
@2
B3
C4"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
min_var
max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
%min_var
&max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
2min_var
3max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:
?min_var
@max_var"
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
&:$@2Adam/dense_68/kernel/m
 :2Adam/dense_68/bias/m
&:$2Adam/dense_69/kernel/m
 :2Adam/dense_69/bias/m
&:$2Adam/dense_70/kernel/m
 :2Adam/dense_70/bias/m
&:$@2Adam/dense_71/kernel/m
 :@2Adam/dense_71/bias/m
&:$@2Adam/dense_68/kernel/v
 :2Adam/dense_68/bias/v
&:$2Adam/dense_69/kernel/v
 :2Adam/dense_69/bias/v
&:$2Adam/dense_70/kernel/v
 :2Adam/dense_70/bias/v
&:$@2Adam/dense_71/kernel/v
 :@2Adam/dense_71/bias/v
�2�
*__inference_model_20_layer_call_fn_4226143
*__inference_model_20_layer_call_fn_4227026
*__inference_model_20_layer_call_fn_4227083
*__inference_model_20_layer_call_fn_4226782�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_model_20_layer_call_and_return_conditional_losses_4227159
E__inference_model_20_layer_call_and_return_conditional_losses_4227391
E__inference_model_20_layer_call_and_return_conditional_losses_4226843
E__inference_model_20_layer_call_and_return_conditional_losses_4226904�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_4225926input_21"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_quantize_layer_20_layer_call_fn_4227400
3__inference_quantize_layer_20_layer_call_fn_4227409�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4227418
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4227439�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_quant_dense_68_layer_call_fn_4227456
0__inference_quant_dense_68_layer_call_fn_4227473�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4227494
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4227551�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_quant_dense_69_layer_call_fn_4227568
0__inference_quant_dense_69_layer_call_fn_4227585�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4227606
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4227663�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_quant_dense_70_layer_call_fn_4227680
0__inference_quant_dense_70_layer_call_fn_4227697�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4227718
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4227775�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_quant_dense_71_layer_call_fn_4227792
0__inference_quant_dense_71_layer_call_fn_4227809�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4227829
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4227885�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_signature_wrapper_4226969input_21"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_4225926�NOP%&Q()R23S56T?@UBC1�.
'�$
"�
input_21���������@
� "?�<
:
quant_dense_71(�%
quant_dense_71���������@�
E__inference_model_20_layer_call_and_return_conditional_losses_4226843~NOP%&Q()R23S56T?@UBC9�6
/�,
"�
input_21���������@
p 

 
� "%�"
�
0���������@
� �
E__inference_model_20_layer_call_and_return_conditional_losses_4226904~NOP%&Q()R23S56T?@UBC9�6
/�,
"�
input_21���������@
p

 
� "%�"
�
0���������@
� �
E__inference_model_20_layer_call_and_return_conditional_losses_4227159|NOP%&Q()R23S56T?@UBC7�4
-�*
 �
inputs���������@
p 

 
� "%�"
�
0���������@
� �
E__inference_model_20_layer_call_and_return_conditional_losses_4227391|NOP%&Q()R23S56T?@UBC7�4
-�*
 �
inputs���������@
p

 
� "%�"
�
0���������@
� �
*__inference_model_20_layer_call_fn_4226143qNOP%&Q()R23S56T?@UBC9�6
/�,
"�
input_21���������@
p 

 
� "����������@�
*__inference_model_20_layer_call_fn_4226782qNOP%&Q()R23S56T?@UBC9�6
/�,
"�
input_21���������@
p

 
� "����������@�
*__inference_model_20_layer_call_fn_4227026oNOP%&Q()R23S56T?@UBC7�4
-�*
 �
inputs���������@
p 

 
� "����������@�
*__inference_model_20_layer_call_fn_4227083oNOP%&Q()R23S56T?@UBC7�4
-�*
 �
inputs���������@
p

 
� "����������@�
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4227494dNO3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������
� �
K__inference_quant_dense_68_layer_call_and_return_conditional_losses_4227551dNO3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������
� �
0__inference_quant_dense_68_layer_call_fn_4227456WNO3�0
)�&
 �
inputs���������@
p 
� "�����������
0__inference_quant_dense_68_layer_call_fn_4227473WNO3�0
)�&
 �
inputs���������@
p
� "�����������
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4227606dP%&Q()3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
K__inference_quant_dense_69_layer_call_and_return_conditional_losses_4227663dP%&Q()3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
0__inference_quant_dense_69_layer_call_fn_4227568WP%&Q()3�0
)�&
 �
inputs���������
p 
� "�����������
0__inference_quant_dense_69_layer_call_fn_4227585WP%&Q()3�0
)�&
 �
inputs���������
p
� "�����������
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4227718dR23S563�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
K__inference_quant_dense_70_layer_call_and_return_conditional_losses_4227775dR23S563�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
0__inference_quant_dense_70_layer_call_fn_4227680WR23S563�0
)�&
 �
inputs���������
p 
� "�����������
0__inference_quant_dense_70_layer_call_fn_4227697WR23S563�0
)�&
 �
inputs���������
p
� "�����������
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4227829dT?@UBC3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������@
� �
K__inference_quant_dense_71_layer_call_and_return_conditional_losses_4227885dT?@UBC3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������@
� �
0__inference_quant_dense_71_layer_call_fn_4227792WT?@UBC3�0
)�&
 �
inputs���������
p 
� "����������@�
0__inference_quant_dense_71_layer_call_fn_4227809WT?@UBC3�0
)�&
 �
inputs���������
p
� "����������@�
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4227418`3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
N__inference_quantize_layer_20_layer_call_and_return_conditional_losses_4227439`3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
3__inference_quantize_layer_20_layer_call_fn_4227400S3�0
)�&
 �
inputs���������@
p 
� "����������@�
3__inference_quantize_layer_20_layer_call_fn_4227409S3�0
)�&
 �
inputs���������@
p
� "����������@�
%__inference_signature_wrapper_4226969�NOP%&Q()R23S56T?@UBC=�:
� 
3�0
.
input_21"�
input_21���������@"?�<
:
quant_dense_71(�%
quant_dense_71���������@