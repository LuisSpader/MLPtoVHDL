�8
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
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��4
�
'quantize_layer_15/quantize_layer_15_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'quantize_layer_15/quantize_layer_15_min
�
;quantize_layer_15/quantize_layer_15_min/Read/ReadVariableOpReadVariableOp'quantize_layer_15/quantize_layer_15_min*
_output_shapes
: *
dtype0
�
'quantize_layer_15/quantize_layer_15_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'quantize_layer_15/quantize_layer_15_max
�
;quantize_layer_15/quantize_layer_15_max/Read/ReadVariableOpReadVariableOp'quantize_layer_15/quantize_layer_15_max*
_output_shapes
: *
dtype0
�
 quantize_layer_15/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quantize_layer_15/optimizer_step
�
4quantize_layer_15/optimizer_step/Read/ReadVariableOpReadVariableOp quantize_layer_15/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_53/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_53/optimizer_step
�
1quant_dense_53/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_53/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_53/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_53/kernel_min

-quant_dense_53/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_53/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_53/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_53/kernel_max

-quant_dense_53/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_53/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_53/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_53/post_activation_min
�
6quant_dense_53/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_53/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_53/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_53/post_activation_max
�
6quant_dense_53/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_53/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_54/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_54/optimizer_step
�
1quant_dense_54/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_54/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_54/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_54/kernel_min

-quant_dense_54/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_54/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_54/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_54/kernel_max

-quant_dense_54/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_54/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_54/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_54/post_activation_min
�
6quant_dense_54/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_54/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_54/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_54/post_activation_max
�
6quant_dense_54/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_54/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_55/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_55/optimizer_step
�
1quant_dense_55/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_55/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_55/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_55/kernel_min

-quant_dense_55/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_55/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_55/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_55/kernel_max

-quant_dense_55/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_55/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_55/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_55/post_activation_min
�
6quant_dense_55/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_55/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_55/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_55/post_activation_max
�
6quant_dense_55/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_55/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_56/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_56/optimizer_step
�
1quant_dense_56/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_56/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_56/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_56/kernel_min

-quant_dense_56/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_56/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_56/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_56/kernel_max

-quant_dense_56/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_56/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_56/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_56/post_activation_min
�
6quant_dense_56/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_56/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_56/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_56/post_activation_max
�
6quant_dense_56/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_56/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_57/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_57/optimizer_step
�
1quant_dense_57/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_57/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_57/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_57/kernel_min

-quant_dense_57/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_57/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_57/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_57/kernel_max

-quant_dense_57/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_57/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_57/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_57/post_activation_min
�
6quant_dense_57/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_57/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_57/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_57/post_activation_max
�
6quant_dense_57/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_57/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_58/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_58/optimizer_step
�
1quant_dense_58/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_58/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_58/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_58/kernel_min

-quant_dense_58/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_58/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_58/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_58/kernel_max

-quant_dense_58/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_58/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_58/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_58/post_activation_min
�
6quant_dense_58/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_58/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_58/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_58/post_activation_max
�
6quant_dense_58/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_58/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_59/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_59/optimizer_step
�
1quant_dense_59/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_59/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_59/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_59/kernel_min

-quant_dense_59/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_59/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_59/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_59/kernel_max

-quant_dense_59/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_59/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_59/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_59/post_activation_min
�
6quant_dense_59/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_59/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_59/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_59/post_activation_max
�
6quant_dense_59/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_59/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_60/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_60/optimizer_step
�
1quant_dense_60/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_60/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_60/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_60/kernel_min

-quant_dense_60/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_60/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_60/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_60/kernel_max

-quant_dense_60/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_60/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_60/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_60/post_activation_min
�
6quant_dense_60/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_60/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_60/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_60/post_activation_max
�
6quant_dense_60/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_60/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_61/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_61/optimizer_step
�
1quant_dense_61/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_61/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_61/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_61/kernel_min

-quant_dense_61/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_61/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_61/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_61/kernel_max

-quant_dense_61/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_61/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_61/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_61/post_activation_min
�
6quant_dense_61/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_61/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_61/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_61/post_activation_max
�
6quant_dense_61/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_61/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_62/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_62/optimizer_step
�
1quant_dense_62/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_62/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_62/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_62/kernel_min

-quant_dense_62/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_62/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_62/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_62/kernel_max

-quant_dense_62/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_62/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_62/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_62/post_activation_min
�
6quant_dense_62/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_62/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_62/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_62/post_activation_max
�
6quant_dense_62/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_62/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_63/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_63/optimizer_step
�
1quant_dense_63/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_63/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_63/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_63/kernel_min

-quant_dense_63/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_63/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_63/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_63/kernel_max

-quant_dense_63/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_63/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_63/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_63/post_activation_min
�
6quant_dense_63/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_63/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_63/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_63/post_activation_max
�
6quant_dense_63/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_63/post_activation_max*
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
dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_53/kernel
s
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel*
_output_shapes

:@@*
dtype0
r
dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_53/bias
k
!dense_53/bias/Read/ReadVariableOpReadVariableOpdense_53/bias*
_output_shapes
:@*
dtype0
z
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@2* 
shared_namedense_54/kernel
s
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel*
_output_shapes

:@2*
dtype0
r
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_54/bias
k
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes
:2*
dtype0
z
dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 * 
shared_namedense_55/kernel
s
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel*
_output_shapes

:2 *
dtype0
r
dense_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_55/bias
k
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
_output_shapes
: *
dtype0
z
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_56/kernel
s
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel*
_output_shapes

: *
dtype0
r
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_56/bias
k
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes
:*
dtype0
z
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_57/kernel
s
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes

:*
dtype0
r
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes
:*
dtype0
z
dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_58/kernel
s
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel*
_output_shapes

:*
dtype0
r
dense_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_58/bias
k
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes
:*
dtype0
z
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_59/kernel
s
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes

:*
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
:*
dtype0
z
dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_60/kernel
s
#dense_60/kernel/Read/ReadVariableOpReadVariableOpdense_60/kernel*
_output_shapes

:*
dtype0
r
dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_60/bias
k
!dense_60/bias/Read/ReadVariableOpReadVariableOpdense_60/bias*
_output_shapes
:*
dtype0
z
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_61/kernel
s
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel*
_output_shapes

: *
dtype0
r
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_61/bias
k
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
_output_shapes
: *
dtype0
z
dense_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_62/kernel
s
#dense_62/kernel/Read/ReadVariableOpReadVariableOpdense_62/kernel*
_output_shapes

:  *
dtype0
r
dense_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_62/bias
k
!dense_62/bias/Read/ReadVariableOpReadVariableOpdense_62/bias*
_output_shapes
: *
dtype0
z
dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_63/kernel
s
#dense_63/kernel/Read/ReadVariableOpReadVariableOpdense_63/kernel*
_output_shapes

: @*
dtype0
r
dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_63/bias
k
!dense_63/bias/Read/ReadVariableOpReadVariableOpdense_63/bias*
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
Adam/dense_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_53/kernel/m
�
*Adam/dense_53/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_53/kernel/m*
_output_shapes

:@@*
dtype0
�
Adam/dense_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_53/bias/m
y
(Adam/dense_53/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_53/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@2*'
shared_nameAdam/dense_54/kernel/m
�
*Adam/dense_54/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/m*
_output_shapes

:@2*
dtype0
�
Adam/dense_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_54/bias/m
y
(Adam/dense_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 *'
shared_nameAdam/dense_55/kernel/m
�
*Adam/dense_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/m*
_output_shapes

:2 *
dtype0
�
Adam/dense_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_55/bias/m
y
(Adam/dense_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_56/kernel/m
�
*Adam/dense_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/m*
_output_shapes

: *
dtype0
�
Adam/dense_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_56/bias/m
y
(Adam/dense_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_57/kernel/m
�
*Adam/dense_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_57/bias/m
y
(Adam/dense_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_58/kernel/m
�
*Adam/dense_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_58/bias/m
y
(Adam/dense_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_59/kernel/m
�
*Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_59/bias/m
y
(Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_60/kernel/m
�
*Adam/dense_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_60/bias/m
y
(Adam/dense_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_61/kernel/m
�
*Adam/dense_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/m*
_output_shapes

: *
dtype0
�
Adam/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_61/bias/m
y
(Adam/dense_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_62/kernel/m
�
*Adam/dense_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/m*
_output_shapes

:  *
dtype0
�
Adam/dense_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_62/bias/m
y
(Adam/dense_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/dense_63/kernel/m
�
*Adam/dense_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/m*
_output_shapes

: @*
dtype0
�
Adam/dense_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_63/bias/m
y
(Adam/dense_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_53/kernel/v
�
*Adam/dense_53/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_53/kernel/v*
_output_shapes

:@@*
dtype0
�
Adam/dense_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_53/bias/v
y
(Adam/dense_53/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_53/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@2*'
shared_nameAdam/dense_54/kernel/v
�
*Adam/dense_54/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/v*
_output_shapes

:@2*
dtype0
�
Adam/dense_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_54/bias/v
y
(Adam/dense_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 *'
shared_nameAdam/dense_55/kernel/v
�
*Adam/dense_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/v*
_output_shapes

:2 *
dtype0
�
Adam/dense_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_55/bias/v
y
(Adam/dense_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_56/kernel/v
�
*Adam/dense_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/v*
_output_shapes

: *
dtype0
�
Adam/dense_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_56/bias/v
y
(Adam/dense_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_57/kernel/v
�
*Adam/dense_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_57/bias/v
y
(Adam/dense_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_58/kernel/v
�
*Adam/dense_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_58/bias/v
y
(Adam/dense_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_59/kernel/v
�
*Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_59/bias/v
y
(Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_60/kernel/v
�
*Adam/dense_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_60/bias/v
y
(Adam/dense_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_61/kernel/v
�
*Adam/dense_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/v*
_output_shapes

: *
dtype0
�
Adam/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_61/bias/v
y
(Adam/dense_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_62/kernel/v
�
*Adam/dense_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/v*
_output_shapes

:  *
dtype0
�
Adam/dense_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_62/bias/v
y
(Adam/dense_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/dense_63/kernel/v
�
*Adam/dense_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/v*
_output_shapes

: @*
dtype0
�
Adam/dense_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_63/bias/v
y
(Adam/dense_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
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
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
�
quantize_layer_15_min
quantize_layer_15_max
quantizer_vars
optimizer_step
	variables
trainable_variables
regularization_losses
	keras_api
�
	layer
optimizer_step
_weight_vars

kernel_min
 
kernel_max
!_quantize_activations
"post_activation_min
#post_activation_max
$_output_quantizers
%	variables
&trainable_variables
'regularization_losses
(	keras_api
�
	)layer
*optimizer_step
+_weight_vars
,
kernel_min
-
kernel_max
._quantize_activations
/post_activation_min
0post_activation_max
1_output_quantizers
2	variables
3trainable_variables
4regularization_losses
5	keras_api
�
	6layer
7optimizer_step
8_weight_vars
9
kernel_min
:
kernel_max
;_quantize_activations
<post_activation_min
=post_activation_max
>_output_quantizers
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
�
	Clayer
Doptimizer_step
E_weight_vars
F
kernel_min
G
kernel_max
H_quantize_activations
Ipost_activation_min
Jpost_activation_max
K_output_quantizers
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
�
	Player
Qoptimizer_step
R_weight_vars
S
kernel_min
T
kernel_max
U_quantize_activations
Vpost_activation_min
Wpost_activation_max
X_output_quantizers
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
�
	]layer
^optimizer_step
__weight_vars
`
kernel_min
a
kernel_max
b_quantize_activations
cpost_activation_min
dpost_activation_max
e_output_quantizers
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
�
	jlayer
koptimizer_step
l_weight_vars
m
kernel_min
n
kernel_max
o_quantize_activations
ppost_activation_min
qpost_activation_max
r_output_quantizers
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
�
	wlayer
xoptimizer_step
y_weight_vars
z
kernel_min
{
kernel_max
|_quantize_activations
}post_activation_min
~post_activation_max
_output_quantizers
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�post_activation_min
�post_activation_max
�_output_quantizers
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�post_activation_min
�post_activation_max
�_output_quantizers
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�post_activation_min
�post_activation_max
�_output_quantizers
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�
�
0
1
2
�3
�4
5
6
 7
"8
#9
�10
�11
*12
,13
-14
/15
016
�17
�18
719
920
:21
<22
=23
�24
�25
D26
F27
G28
I29
J30
�31
�32
Q33
S34
T35
V36
W37
�38
�39
^40
`41
a42
c43
d44
�45
�46
k47
m48
n49
p50
q51
�52
�53
x54
z55
{56
}57
~58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 
��
VARIABLE_VALUE'quantize_layer_15/quantize_layer_15_minElayer_with_weights-0/quantize_layer_15_min/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'quantize_layer_15/quantize_layer_15_maxElayer_with_weights-0/quantize_layer_15_max/.ATTRIBUTES/VARIABLE_VALUE

min_var
max_var
tr
VARIABLE_VALUE quantize_layer_15/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_53/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_53/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_53/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_53/post_activation_minClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_53/post_activation_maxClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
�0
�1
2
3
 4
"5
#6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_54/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_54/kernel_min:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_54/kernel_max:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_54/post_activation_minClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_54/post_activation_maxClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
�0
�1
*2
,3
-4
/5
06

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_55/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_55/kernel_min:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_55/kernel_max:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_55/post_activation_minClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_55/post_activation_maxClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
�0
�1
72
93
:4
<5
=6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_56/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_56/kernel_min:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_56/kernel_max:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_56/post_activation_minClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_56/post_activation_maxClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
�0
�1
D2
F3
G4
I5
J6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_57/optimizer_step>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_57/kernel_min:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_57/kernel_max:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_57/post_activation_minClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_57/post_activation_maxClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
�0
�1
Q2
S3
T4
V5
W6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_58/optimizer_step>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_58/kernel_min:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_58/kernel_max:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_58/post_activation_minClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_58/post_activation_maxClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
�0
�1
^2
`3
a4
c5
d6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_59/optimizer_step>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_59/kernel_min:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_59/kernel_max:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_59/post_activation_minClayer_with_weights-7/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_59/post_activation_maxClayer_with_weights-7/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
�0
�1
k2
m3
n4
p5
q6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_60/optimizer_step>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_60/kernel_min:layer_with_weights-8/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_60/kernel_max:layer_with_weights-8/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_60/post_activation_minClayer_with_weights-8/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_60/post_activation_maxClayer_with_weights-8/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
3
�0
�1
x2
z3
{4
}5
~6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
qo
VARIABLE_VALUEquant_dense_61/optimizer_step>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
ig
VARIABLE_VALUEquant_dense_61/kernel_min:layer_with_weights-9/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_61/kernel_max:layer_with_weights-9/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_61/post_activation_minClayer_with_weights-9/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_61/post_activation_maxClayer_with_weights-9/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
8
�0
�1
�2
�3
�4
�5
�6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
rp
VARIABLE_VALUEquant_dense_62/optimizer_step?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
jh
VARIABLE_VALUEquant_dense_62/kernel_min;layer_with_weights-10/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEquant_dense_62/kernel_max;layer_with_weights-10/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
|z
VARIABLE_VALUE"quant_dense_62/post_activation_minDlayer_with_weights-10/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE"quant_dense_62/post_activation_maxDlayer_with_weights-10/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
8
�0
�1
�2
�3
�4
�5
�6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
rp
VARIABLE_VALUEquant_dense_63/optimizer_step?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
jh
VARIABLE_VALUEquant_dense_63/kernel_min;layer_with_weights-11/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEquant_dense_63/kernel_max;layer_with_weights-11/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
|z
VARIABLE_VALUE"quant_dense_63/post_activation_minDlayer_with_weights-11/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE"quant_dense_63/post_activation_maxDlayer_with_weights-11/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
8
�0
�1
�2
�3
�4
�5
�6

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
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
VARIABLE_VALUEdense_53/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_53/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_54/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_54/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_55/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_55/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_56/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_56/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_57/kernel'variables/31/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_57/bias'variables/32/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_58/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_58/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_59/kernel'variables/45/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_59/bias'variables/46/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_60/kernel'variables/52/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_60/bias'variables/53/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_61/kernel'variables/59/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_61/bias'variables/60/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_62/kernel'variables/66/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_62/bias'variables/67/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_63/kernel'variables/73/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_63/bias'variables/74/.ATTRIBUTES/VARIABLE_VALUE
�
0
1
2
3
4
 5
"6
#7
*8
,9
-10
/11
012
713
914
:15
<16
=17
D18
F19
G20
I21
J22
Q23
S24
T25
V26
W27
^28
`29
a30
c31
d32
k33
m34
n35
p36
q37
x38
z39
{40
}41
~42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
^
0
1
2
3
4
5
6
7
	8

9
10
11
12

�0
 
 

0
1
2
 
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
#
0
1
 2
"3
#4

0
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
#
*0
,1
-2
/3
04

)0
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
#
70
91
:2
<3
=4

60
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
#
D0
F1
G2
I3
J4

C0
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
#
Q0
S1
T2
V3
W4

P0
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
#
^0
`1
a2
c3
d4

]0
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
#
k0
m1
n2
p3
q4

j0
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
#
x0
z1
{2
}3
~4

w0
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
(
�0
�1
�2
�3
�4

�0
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
(
�0
�1
�2
�3
�4

�0
 
 
 

�0

�0
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

�0
�2
(
�0
�1
�2
�3
�4

�0
 
 
 
8

�total

�count
�	variables
�	keras_api
 
 
 
 
 

min_var
 max_var
 
 
 
 
 

,min_var
-max_var
 
 
 
 
 

9min_var
:max_var
 
 
 
 
 

Fmin_var
Gmax_var
 
 
 
 
 

Smin_var
Tmax_var
 
 
 
 
 

`min_var
amax_var
 
 
 
 
 

mmin_var
nmax_var
 
 
 
 
 

zmin_var
{max_var
 
 
 
 
 

�min_var
�max_var
 
 
 
 
 

�min_var
�max_var
 
 
 
 
 

�min_var
�max_var
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
nl
VARIABLE_VALUEAdam/dense_53/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_53/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_54/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_54/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_55/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_55/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_56/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_56/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_57/kernel/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_57/bias/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_58/kernel/mCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_58/bias/mCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_59/kernel/mCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_59/bias/mCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_60/kernel/mCvariables/52/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_60/bias/mCvariables/53/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_61/kernel/mCvariables/59/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_61/bias/mCvariables/60/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_62/kernel/mCvariables/66/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_62/bias/mCvariables/67/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_63/kernel/mCvariables/73/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_63/bias/mCvariables/74/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_53/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_53/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_54/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_54/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_55/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_55/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_56/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_56/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_57/kernel/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_57/bias/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_58/kernel/vCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_58/bias/vCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_59/kernel/vCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_59/bias/vCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_60/kernel/vCvariables/52/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_60/bias/vCvariables/53/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_61/kernel/vCvariables/59/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_61/bias/vCvariables/60/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_62/kernel/vCvariables/66/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_62/bias/vCvariables/67/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_63/kernel/vCvariables/73/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_63/bias/vCvariables/74/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_16Placeholder*'
_output_shapes
:���������@*
dtype0*
shape:���������@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_16'quantize_layer_15/quantize_layer_15_min'quantize_layer_15/quantize_layer_15_maxdense_53/kernelquant_dense_53/kernel_minquant_dense_53/kernel_maxdense_53/bias"quant_dense_53/post_activation_min"quant_dense_53/post_activation_maxdense_54/kernelquant_dense_54/kernel_minquant_dense_54/kernel_maxdense_54/bias"quant_dense_54/post_activation_min"quant_dense_54/post_activation_maxdense_55/kernelquant_dense_55/kernel_minquant_dense_55/kernel_maxdense_55/bias"quant_dense_55/post_activation_min"quant_dense_55/post_activation_maxdense_56/kernelquant_dense_56/kernel_minquant_dense_56/kernel_maxdense_56/bias"quant_dense_56/post_activation_min"quant_dense_56/post_activation_maxdense_57/kernelquant_dense_57/kernel_minquant_dense_57/kernel_maxdense_57/bias"quant_dense_57/post_activation_min"quant_dense_57/post_activation_maxdense_58/kernelquant_dense_58/kernel_minquant_dense_58/kernel_maxdense_58/bias"quant_dense_58/post_activation_min"quant_dense_58/post_activation_maxdense_59/kernelquant_dense_59/kernel_minquant_dense_59/kernel_maxdense_59/bias"quant_dense_59/post_activation_min"quant_dense_59/post_activation_maxdense_60/kernelquant_dense_60/kernel_minquant_dense_60/kernel_maxdense_60/bias"quant_dense_60/post_activation_min"quant_dense_60/post_activation_maxdense_61/kernelquant_dense_61/kernel_minquant_dense_61/kernel_maxdense_61/bias"quant_dense_61/post_activation_min"quant_dense_61/post_activation_maxdense_62/kernelquant_dense_62/kernel_minquant_dense_62/kernel_maxdense_62/bias"quant_dense_62/post_activation_min"quant_dense_62/post_activation_maxdense_63/kernelquant_dense_63/kernel_minquant_dense_63/kernel_maxdense_63/bias"quant_dense_63/post_activation_min"quant_dense_63/post_activation_max*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_3752913
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�1
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename;quantize_layer_15/quantize_layer_15_min/Read/ReadVariableOp;quantize_layer_15/quantize_layer_15_max/Read/ReadVariableOp4quantize_layer_15/optimizer_step/Read/ReadVariableOp1quant_dense_53/optimizer_step/Read/ReadVariableOp-quant_dense_53/kernel_min/Read/ReadVariableOp-quant_dense_53/kernel_max/Read/ReadVariableOp6quant_dense_53/post_activation_min/Read/ReadVariableOp6quant_dense_53/post_activation_max/Read/ReadVariableOp1quant_dense_54/optimizer_step/Read/ReadVariableOp-quant_dense_54/kernel_min/Read/ReadVariableOp-quant_dense_54/kernel_max/Read/ReadVariableOp6quant_dense_54/post_activation_min/Read/ReadVariableOp6quant_dense_54/post_activation_max/Read/ReadVariableOp1quant_dense_55/optimizer_step/Read/ReadVariableOp-quant_dense_55/kernel_min/Read/ReadVariableOp-quant_dense_55/kernel_max/Read/ReadVariableOp6quant_dense_55/post_activation_min/Read/ReadVariableOp6quant_dense_55/post_activation_max/Read/ReadVariableOp1quant_dense_56/optimizer_step/Read/ReadVariableOp-quant_dense_56/kernel_min/Read/ReadVariableOp-quant_dense_56/kernel_max/Read/ReadVariableOp6quant_dense_56/post_activation_min/Read/ReadVariableOp6quant_dense_56/post_activation_max/Read/ReadVariableOp1quant_dense_57/optimizer_step/Read/ReadVariableOp-quant_dense_57/kernel_min/Read/ReadVariableOp-quant_dense_57/kernel_max/Read/ReadVariableOp6quant_dense_57/post_activation_min/Read/ReadVariableOp6quant_dense_57/post_activation_max/Read/ReadVariableOp1quant_dense_58/optimizer_step/Read/ReadVariableOp-quant_dense_58/kernel_min/Read/ReadVariableOp-quant_dense_58/kernel_max/Read/ReadVariableOp6quant_dense_58/post_activation_min/Read/ReadVariableOp6quant_dense_58/post_activation_max/Read/ReadVariableOp1quant_dense_59/optimizer_step/Read/ReadVariableOp-quant_dense_59/kernel_min/Read/ReadVariableOp-quant_dense_59/kernel_max/Read/ReadVariableOp6quant_dense_59/post_activation_min/Read/ReadVariableOp6quant_dense_59/post_activation_max/Read/ReadVariableOp1quant_dense_60/optimizer_step/Read/ReadVariableOp-quant_dense_60/kernel_min/Read/ReadVariableOp-quant_dense_60/kernel_max/Read/ReadVariableOp6quant_dense_60/post_activation_min/Read/ReadVariableOp6quant_dense_60/post_activation_max/Read/ReadVariableOp1quant_dense_61/optimizer_step/Read/ReadVariableOp-quant_dense_61/kernel_min/Read/ReadVariableOp-quant_dense_61/kernel_max/Read/ReadVariableOp6quant_dense_61/post_activation_min/Read/ReadVariableOp6quant_dense_61/post_activation_max/Read/ReadVariableOp1quant_dense_62/optimizer_step/Read/ReadVariableOp-quant_dense_62/kernel_min/Read/ReadVariableOp-quant_dense_62/kernel_max/Read/ReadVariableOp6quant_dense_62/post_activation_min/Read/ReadVariableOp6quant_dense_62/post_activation_max/Read/ReadVariableOp1quant_dense_63/optimizer_step/Read/ReadVariableOp-quant_dense_63/kernel_min/Read/ReadVariableOp-quant_dense_63/kernel_max/Read/ReadVariableOp6quant_dense_63/post_activation_min/Read/ReadVariableOp6quant_dense_63/post_activation_max/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#dense_53/kernel/Read/ReadVariableOp!dense_53/bias/Read/ReadVariableOp#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOp#dense_55/kernel/Read/ReadVariableOp!dense_55/bias/Read/ReadVariableOp#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOp#dense_58/kernel/Read/ReadVariableOp!dense_58/bias/Read/ReadVariableOp#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOp#dense_60/kernel/Read/ReadVariableOp!dense_60/bias/Read/ReadVariableOp#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOp#dense_62/kernel/Read/ReadVariableOp!dense_62/bias/Read/ReadVariableOp#dense_63/kernel/Read/ReadVariableOp!dense_63/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_53/kernel/m/Read/ReadVariableOp(Adam/dense_53/bias/m/Read/ReadVariableOp*Adam/dense_54/kernel/m/Read/ReadVariableOp(Adam/dense_54/bias/m/Read/ReadVariableOp*Adam/dense_55/kernel/m/Read/ReadVariableOp(Adam/dense_55/bias/m/Read/ReadVariableOp*Adam/dense_56/kernel/m/Read/ReadVariableOp(Adam/dense_56/bias/m/Read/ReadVariableOp*Adam/dense_57/kernel/m/Read/ReadVariableOp(Adam/dense_57/bias/m/Read/ReadVariableOp*Adam/dense_58/kernel/m/Read/ReadVariableOp(Adam/dense_58/bias/m/Read/ReadVariableOp*Adam/dense_59/kernel/m/Read/ReadVariableOp(Adam/dense_59/bias/m/Read/ReadVariableOp*Adam/dense_60/kernel/m/Read/ReadVariableOp(Adam/dense_60/bias/m/Read/ReadVariableOp*Adam/dense_61/kernel/m/Read/ReadVariableOp(Adam/dense_61/bias/m/Read/ReadVariableOp*Adam/dense_62/kernel/m/Read/ReadVariableOp(Adam/dense_62/bias/m/Read/ReadVariableOp*Adam/dense_63/kernel/m/Read/ReadVariableOp(Adam/dense_63/bias/m/Read/ReadVariableOp*Adam/dense_53/kernel/v/Read/ReadVariableOp(Adam/dense_53/bias/v/Read/ReadVariableOp*Adam/dense_54/kernel/v/Read/ReadVariableOp(Adam/dense_54/bias/v/Read/ReadVariableOp*Adam/dense_55/kernel/v/Read/ReadVariableOp(Adam/dense_55/bias/v/Read/ReadVariableOp*Adam/dense_56/kernel/v/Read/ReadVariableOp(Adam/dense_56/bias/v/Read/ReadVariableOp*Adam/dense_57/kernel/v/Read/ReadVariableOp(Adam/dense_57/bias/v/Read/ReadVariableOp*Adam/dense_58/kernel/v/Read/ReadVariableOp(Adam/dense_58/bias/v/Read/ReadVariableOp*Adam/dense_59/kernel/v/Read/ReadVariableOp(Adam/dense_59/bias/v/Read/ReadVariableOp*Adam/dense_60/kernel/v/Read/ReadVariableOp(Adam/dense_60/bias/v/Read/ReadVariableOp*Adam/dense_61/kernel/v/Read/ReadVariableOp(Adam/dense_61/bias/v/Read/ReadVariableOp*Adam/dense_62/kernel/v/Read/ReadVariableOp(Adam/dense_62/bias/v/Read/ReadVariableOp*Adam/dense_63/kernel/v/Read/ReadVariableOp(Adam/dense_63/bias/v/Read/ReadVariableOpConst*�
Tin�
�2�	*
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
 __inference__traced_save_3755687
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename'quantize_layer_15/quantize_layer_15_min'quantize_layer_15/quantize_layer_15_max quantize_layer_15/optimizer_stepquant_dense_53/optimizer_stepquant_dense_53/kernel_minquant_dense_53/kernel_max"quant_dense_53/post_activation_min"quant_dense_53/post_activation_maxquant_dense_54/optimizer_stepquant_dense_54/kernel_minquant_dense_54/kernel_max"quant_dense_54/post_activation_min"quant_dense_54/post_activation_maxquant_dense_55/optimizer_stepquant_dense_55/kernel_minquant_dense_55/kernel_max"quant_dense_55/post_activation_min"quant_dense_55/post_activation_maxquant_dense_56/optimizer_stepquant_dense_56/kernel_minquant_dense_56/kernel_max"quant_dense_56/post_activation_min"quant_dense_56/post_activation_maxquant_dense_57/optimizer_stepquant_dense_57/kernel_minquant_dense_57/kernel_max"quant_dense_57/post_activation_min"quant_dense_57/post_activation_maxquant_dense_58/optimizer_stepquant_dense_58/kernel_minquant_dense_58/kernel_max"quant_dense_58/post_activation_min"quant_dense_58/post_activation_maxquant_dense_59/optimizer_stepquant_dense_59/kernel_minquant_dense_59/kernel_max"quant_dense_59/post_activation_min"quant_dense_59/post_activation_maxquant_dense_60/optimizer_stepquant_dense_60/kernel_minquant_dense_60/kernel_max"quant_dense_60/post_activation_min"quant_dense_60/post_activation_maxquant_dense_61/optimizer_stepquant_dense_61/kernel_minquant_dense_61/kernel_max"quant_dense_61/post_activation_min"quant_dense_61/post_activation_maxquant_dense_62/optimizer_stepquant_dense_62/kernel_minquant_dense_62/kernel_max"quant_dense_62/post_activation_min"quant_dense_62/post_activation_maxquant_dense_63/optimizer_stepquant_dense_63/kernel_minquant_dense_63/kernel_max"quant_dense_63/post_activation_min"quant_dense_63/post_activation_max	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_53/kerneldense_53/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/biasdense_60/kerneldense_60/biasdense_61/kerneldense_61/biasdense_62/kerneldense_62/biasdense_63/kerneldense_63/biastotalcountAdam/dense_53/kernel/mAdam/dense_53/bias/mAdam/dense_54/kernel/mAdam/dense_54/bias/mAdam/dense_55/kernel/mAdam/dense_55/bias/mAdam/dense_56/kernel/mAdam/dense_56/bias/mAdam/dense_57/kernel/mAdam/dense_57/bias/mAdam/dense_58/kernel/mAdam/dense_58/bias/mAdam/dense_59/kernel/mAdam/dense_59/bias/mAdam/dense_60/kernel/mAdam/dense_60/bias/mAdam/dense_61/kernel/mAdam/dense_61/bias/mAdam/dense_62/kernel/mAdam/dense_62/bias/mAdam/dense_63/kernel/mAdam/dense_63/bias/mAdam/dense_53/kernel/vAdam/dense_53/bias/vAdam/dense_54/kernel/vAdam/dense_54/bias/vAdam/dense_55/kernel/vAdam/dense_55/bias/vAdam/dense_56/kernel/vAdam/dense_56/bias/vAdam/dense_57/kernel/vAdam/dense_57/bias/vAdam/dense_58/kernel/vAdam/dense_58/bias/vAdam/dense_59/kernel/vAdam/dense_59/bias/vAdam/dense_60/kernel/vAdam/dense_60/bias/vAdam/dense_61/kernel/vAdam/dense_61/bias/vAdam/dense_62/kernel/vAdam/dense_62/bias/vAdam/dense_63/kernel/vAdam/dense_63/bias/v*�
Tin�
�2�*
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
#__inference__traced_restore_3756090ǫ0
�
�
0__inference_quant_dense_59_layer_call_fn_3754730

inputs
unknown:
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
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3750541o
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
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3751646

inputs=
+lastvaluequant_rank_readvariableop_resource:2 /
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:2 *
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

:2 *
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:2 *
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

:2 *
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

:2 *
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

:2 *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� h
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : : : : : 20
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
:���������2
 
_user_specified_nameinputs
�T
�
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3750910

inputs=
+lastvaluequant_rank_readvariableop_resource: @/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: @*
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

: @*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: @*
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

: @*
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

: @*
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

: @*
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
:��������� : : : : : : 20
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
:��������� 
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3751278

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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

:*
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
:���������: : : : : : 20
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
:���������
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3751554

inputs=
+lastvaluequant_rank_readvariableop_resource: /
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
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

: *
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
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

: *
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

: *
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

: *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 20
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
:��������� 
 
_user_specified_nameinputs
�"
�
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3754041

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
0__inference_quant_dense_61_layer_call_fn_3754971

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
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
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3751094o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Y
�
E__inference_model_15_layer_call_and_return_conditional_losses_3752612
input_16#
quantize_layer_15_3752463: #
quantize_layer_15_3752465: (
quant_dense_53_3752468:@@ 
quant_dense_53_3752470:  
quant_dense_53_3752472: $
quant_dense_53_3752474:@ 
quant_dense_53_3752476:  
quant_dense_53_3752478: (
quant_dense_54_3752481:@2 
quant_dense_54_3752483:  
quant_dense_54_3752485: $
quant_dense_54_3752487:2 
quant_dense_54_3752489:  
quant_dense_54_3752491: (
quant_dense_55_3752494:2  
quant_dense_55_3752496:  
quant_dense_55_3752498: $
quant_dense_55_3752500:  
quant_dense_55_3752502:  
quant_dense_55_3752504: (
quant_dense_56_3752507:  
quant_dense_56_3752509:  
quant_dense_56_3752511: $
quant_dense_56_3752513: 
quant_dense_56_3752515:  
quant_dense_56_3752517: (
quant_dense_57_3752520: 
quant_dense_57_3752522:  
quant_dense_57_3752524: $
quant_dense_57_3752526: 
quant_dense_57_3752528:  
quant_dense_57_3752530: (
quant_dense_58_3752533: 
quant_dense_58_3752535:  
quant_dense_58_3752537: $
quant_dense_58_3752539: 
quant_dense_58_3752541:  
quant_dense_58_3752543: (
quant_dense_59_3752546: 
quant_dense_59_3752548:  
quant_dense_59_3752550: $
quant_dense_59_3752552: 
quant_dense_59_3752554:  
quant_dense_59_3752556: (
quant_dense_60_3752559: 
quant_dense_60_3752561:  
quant_dense_60_3752563: $
quant_dense_60_3752565: 
quant_dense_60_3752567:  
quant_dense_60_3752569: (
quant_dense_61_3752572:  
quant_dense_61_3752574:  
quant_dense_61_3752576: $
quant_dense_61_3752578:  
quant_dense_61_3752580:  
quant_dense_61_3752582: (
quant_dense_62_3752585:   
quant_dense_62_3752587:  
quant_dense_62_3752589: $
quant_dense_62_3752591:  
quant_dense_62_3752593:  
quant_dense_62_3752595: (
quant_dense_63_3752598: @ 
quant_dense_63_3752600:  
quant_dense_63_3752602: $
quant_dense_63_3752604:@ 
quant_dense_63_3752606:  
quant_dense_63_3752608: 
identity��&quant_dense_53/StatefulPartitionedCall�&quant_dense_54/StatefulPartitionedCall�&quant_dense_55/StatefulPartitionedCall�&quant_dense_56/StatefulPartitionedCall�&quant_dense_57/StatefulPartitionedCall�&quant_dense_58/StatefulPartitionedCall�&quant_dense_59/StatefulPartitionedCall�&quant_dense_60/StatefulPartitionedCall�&quant_dense_61/StatefulPartitionedCall�&quant_dense_62/StatefulPartitionedCall�&quant_dense_63/StatefulPartitionedCall�)quantize_layer_15/StatefulPartitionedCall�
)quantize_layer_15/StatefulPartitionedCallStatefulPartitionedCallinput_16quantize_layer_15_3752463quantize_layer_15_3752465*
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
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3750304�
&quant_dense_53/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_15/StatefulPartitionedCall:output:0quant_dense_53_3752468quant_dense_53_3752470quant_dense_53_3752472quant_dense_53_3752474quant_dense_53_3752476quant_dense_53_3752478*
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
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3750331�
&quant_dense_54/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_53/StatefulPartitionedCall:output:0quant_dense_54_3752481quant_dense_54_3752483quant_dense_54_3752485quant_dense_54_3752487quant_dense_54_3752489quant_dense_54_3752491*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3750366�
&quant_dense_55/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_54/StatefulPartitionedCall:output:0quant_dense_55_3752494quant_dense_55_3752496quant_dense_55_3752498quant_dense_55_3752500quant_dense_55_3752502quant_dense_55_3752504*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3750401�
&quant_dense_56/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_55/StatefulPartitionedCall:output:0quant_dense_56_3752507quant_dense_56_3752509quant_dense_56_3752511quant_dense_56_3752513quant_dense_56_3752515quant_dense_56_3752517*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3750436�
&quant_dense_57/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_56/StatefulPartitionedCall:output:0quant_dense_57_3752520quant_dense_57_3752522quant_dense_57_3752524quant_dense_57_3752526quant_dense_57_3752528quant_dense_57_3752530*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3750471�
&quant_dense_58/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_57/StatefulPartitionedCall:output:0quant_dense_58_3752533quant_dense_58_3752535quant_dense_58_3752537quant_dense_58_3752539quant_dense_58_3752541quant_dense_58_3752543*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3750506�
&quant_dense_59/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_58/StatefulPartitionedCall:output:0quant_dense_59_3752546quant_dense_59_3752548quant_dense_59_3752550quant_dense_59_3752552quant_dense_59_3752554quant_dense_59_3752556*
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
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3750541�
&quant_dense_60/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_59/StatefulPartitionedCall:output:0quant_dense_60_3752559quant_dense_60_3752561quant_dense_60_3752563quant_dense_60_3752565quant_dense_60_3752567quant_dense_60_3752569*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3750576�
&quant_dense_61/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_60/StatefulPartitionedCall:output:0quant_dense_61_3752572quant_dense_61_3752574quant_dense_61_3752576quant_dense_61_3752578quant_dense_61_3752580quant_dense_61_3752582*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3750611�
&quant_dense_62/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_61/StatefulPartitionedCall:output:0quant_dense_62_3752585quant_dense_62_3752587quant_dense_62_3752589quant_dense_62_3752591quant_dense_62_3752593quant_dense_62_3752595*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3750646�
&quant_dense_63/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_62/StatefulPartitionedCall:output:0quant_dense_63_3752598quant_dense_63_3752600quant_dense_63_3752602quant_dense_63_3752604quant_dense_63_3752606quant_dense_63_3752608*
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
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3750680~
IdentityIdentity/quant_dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_53/StatefulPartitionedCall'^quant_dense_54/StatefulPartitionedCall'^quant_dense_55/StatefulPartitionedCall'^quant_dense_56/StatefulPartitionedCall'^quant_dense_57/StatefulPartitionedCall'^quant_dense_58/StatefulPartitionedCall'^quant_dense_59/StatefulPartitionedCall'^quant_dense_60/StatefulPartitionedCall'^quant_dense_61/StatefulPartitionedCall'^quant_dense_62/StatefulPartitionedCall'^quant_dense_63/StatefulPartitionedCall*^quantize_layer_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&quant_dense_53/StatefulPartitionedCall&quant_dense_53/StatefulPartitionedCall2P
&quant_dense_54/StatefulPartitionedCall&quant_dense_54/StatefulPartitionedCall2P
&quant_dense_55/StatefulPartitionedCall&quant_dense_55/StatefulPartitionedCall2P
&quant_dense_56/StatefulPartitionedCall&quant_dense_56/StatefulPartitionedCall2P
&quant_dense_57/StatefulPartitionedCall&quant_dense_57/StatefulPartitionedCall2P
&quant_dense_58/StatefulPartitionedCall&quant_dense_58/StatefulPartitionedCall2P
&quant_dense_59/StatefulPartitionedCall&quant_dense_59/StatefulPartitionedCall2P
&quant_dense_60/StatefulPartitionedCall&quant_dense_60/StatefulPartitionedCall2P
&quant_dense_61/StatefulPartitionedCall&quant_dense_61/StatefulPartitionedCall2P
&quant_dense_62/StatefulPartitionedCall&quant_dense_62/StatefulPartitionedCall2P
&quant_dense_63/StatefulPartitionedCall&quant_dense_63/StatefulPartitionedCall2V
)quantize_layer_15/StatefulPartitionedCall)quantize_layer_15/StatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_16
�U
�
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3754601

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
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
:���������
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3751186

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
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
0__inference_quant_dense_58_layer_call_fn_3754618

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
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
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3750506o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3750331

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@@J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@@*
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

:@@*
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
:���������@P
ReluReluBiasAdd:output:0*
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
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
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
�
�
3__inference_quantize_layer_15_layer_call_fn_3754011

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
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3751878o
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
�
�
0__inference_quant_dense_57_layer_call_fn_3754523

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
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
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3751462o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_56_layer_call_fn_3754411

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2:
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
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3751554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_55_layer_call_fn_3754299

inputs
unknown:2 
	unknown_0: 
	unknown_1: 
	unknown_2: 
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
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3751646o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3751738

inputs=
+lastvaluequant_rank_readvariableop_resource:@2/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:2@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@2*
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

:@2*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@2*
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

:@2*
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

:@2*
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

:@2*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2h
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
:���������2�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������2�
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
� 
�
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3750576

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
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
�U
�
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3754377

inputs=
+lastvaluequant_rank_readvariableop_resource:2 /
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:2 *
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

:2 *
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:2 *
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

:2 *
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

:2 *
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

:2 *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� h
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : : : : : 20
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
:���������2
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_57_layer_call_fn_3754506

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
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
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3750471o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3754096

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@@J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@@*
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

:@@*
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
:���������@P
ReluReluBiasAdd:output:0*
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
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
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
�U
�
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3754713

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
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
:���������
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3754768

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
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

:*
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
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3754208

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@2J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:2K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@2*
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

:@2*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2�
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
:���������2�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������2�
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
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3754432

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: *
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

: *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3754880

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
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
�
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3750436

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: *
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

: *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_model_15_layer_call_fn_3752460
input_16
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
	unknown_7:@2
	unknown_8: 
	unknown_9: 

unknown_10:2

unknown_11: 

unknown_12: 

unknown_13:2 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:

unknown_23: 

unknown_24: 

unknown_25:

unknown_26: 

unknown_27: 

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:

unknown_32: 

unknown_33: 

unknown_34:

unknown_35: 

unknown_36: 

unknown_37:

unknown_38: 

unknown_39: 

unknown_40:

unknown_41: 

unknown_42: 

unknown_43:

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: 

unknown_53: 

unknown_54: 

unknown_55:  

unknown_56: 

unknown_57: 

unknown_58: 

unknown_59: 

unknown_60: 

unknown_61: @

unknown_62: 

unknown_63: 

unknown_64:@

unknown_65: 

unknown_66: 
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*8
_read_only_resource_inputs
	!$'*-0369<?B*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_15_layer_call_and_return_conditional_losses_3752180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_16
� 
�
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3754992

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: *
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

: *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_63_layer_call_fn_3755178

inputs
unknown: @
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
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3750680o
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
:��������� : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3750646

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:  J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:  *
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

:  *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_58_layer_call_fn_3754635

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
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
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3751370o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3751370

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
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
:���������
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_55_layer_call_fn_3754282

inputs
unknown:2 
	unknown_0: 
	unknown_1: 
	unknown_2: 
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
:��������� *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3750401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
��	
�u
E__inference_model_15_layer_call_and_return_conditional_losses_3753993

inputsM
Cquantize_layer_15_allvaluesquantize_minimum_readvariableop_resource: M
Cquantize_layer_15_allvaluesquantize_maximum_readvariableop_resource: L
:quant_dense_53_lastvaluequant_rank_readvariableop_resource:@@>
4quant_dense_53_lastvaluequant_assignminlast_resource: >
4quant_dense_53_lastvaluequant_assignmaxlast_resource: <
.quant_dense_53_biasadd_readvariableop_resource:@O
Equant_dense_53_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_53_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_54_lastvaluequant_rank_readvariableop_resource:@2>
4quant_dense_54_lastvaluequant_assignminlast_resource: >
4quant_dense_54_lastvaluequant_assignmaxlast_resource: <
.quant_dense_54_biasadd_readvariableop_resource:2O
Equant_dense_54_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_54_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_55_lastvaluequant_rank_readvariableop_resource:2 >
4quant_dense_55_lastvaluequant_assignminlast_resource: >
4quant_dense_55_lastvaluequant_assignmaxlast_resource: <
.quant_dense_55_biasadd_readvariableop_resource: O
Equant_dense_55_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_55_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_56_lastvaluequant_rank_readvariableop_resource: >
4quant_dense_56_lastvaluequant_assignminlast_resource: >
4quant_dense_56_lastvaluequant_assignmaxlast_resource: <
.quant_dense_56_biasadd_readvariableop_resource:O
Equant_dense_56_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_56_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_57_lastvaluequant_rank_readvariableop_resource:>
4quant_dense_57_lastvaluequant_assignminlast_resource: >
4quant_dense_57_lastvaluequant_assignmaxlast_resource: <
.quant_dense_57_biasadd_readvariableop_resource:O
Equant_dense_57_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_57_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_58_lastvaluequant_rank_readvariableop_resource:>
4quant_dense_58_lastvaluequant_assignminlast_resource: >
4quant_dense_58_lastvaluequant_assignmaxlast_resource: <
.quant_dense_58_biasadd_readvariableop_resource:O
Equant_dense_58_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_58_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_59_lastvaluequant_rank_readvariableop_resource:>
4quant_dense_59_lastvaluequant_assignminlast_resource: >
4quant_dense_59_lastvaluequant_assignmaxlast_resource: <
.quant_dense_59_biasadd_readvariableop_resource:O
Equant_dense_59_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_59_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_60_lastvaluequant_rank_readvariableop_resource:>
4quant_dense_60_lastvaluequant_assignminlast_resource: >
4quant_dense_60_lastvaluequant_assignmaxlast_resource: <
.quant_dense_60_biasadd_readvariableop_resource:O
Equant_dense_60_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_60_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_61_lastvaluequant_rank_readvariableop_resource: >
4quant_dense_61_lastvaluequant_assignminlast_resource: >
4quant_dense_61_lastvaluequant_assignmaxlast_resource: <
.quant_dense_61_biasadd_readvariableop_resource: O
Equant_dense_61_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_61_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_62_lastvaluequant_rank_readvariableop_resource:  >
4quant_dense_62_lastvaluequant_assignminlast_resource: >
4quant_dense_62_lastvaluequant_assignmaxlast_resource: <
.quant_dense_62_biasadd_readvariableop_resource: O
Equant_dense_62_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_62_movingavgquantize_assignmaxema_readvariableop_resource: L
:quant_dense_63_lastvaluequant_rank_readvariableop_resource: @>
4quant_dense_63_lastvaluequant_assignminlast_resource: >
4quant_dense_63_lastvaluequant_assignmaxlast_resource: <
.quant_dense_63_biasadd_readvariableop_resource:@O
Equant_dense_63_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_63_movingavgquantize_assignmaxema_readvariableop_resource: 
identity��%quant_dense_53/BiasAdd/ReadVariableOp�+quant_dense_53/LastValueQuant/AssignMaxLast�+quant_dense_53/LastValueQuant/AssignMinLast�5quant_dense_53/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_53/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_53/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_53/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_53/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_53/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_54/BiasAdd/ReadVariableOp�+quant_dense_54/LastValueQuant/AssignMaxLast�+quant_dense_54/LastValueQuant/AssignMinLast�5quant_dense_54/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_54/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_54/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_54/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_54/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_54/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_55/BiasAdd/ReadVariableOp�+quant_dense_55/LastValueQuant/AssignMaxLast�+quant_dense_55/LastValueQuant/AssignMinLast�5quant_dense_55/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_55/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_55/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_55/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_55/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_55/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_56/BiasAdd/ReadVariableOp�+quant_dense_56/LastValueQuant/AssignMaxLast�+quant_dense_56/LastValueQuant/AssignMinLast�5quant_dense_56/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_56/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_56/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_56/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_56/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_56/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_57/BiasAdd/ReadVariableOp�+quant_dense_57/LastValueQuant/AssignMaxLast�+quant_dense_57/LastValueQuant/AssignMinLast�5quant_dense_57/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_57/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_57/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_57/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_57/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_57/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_58/BiasAdd/ReadVariableOp�+quant_dense_58/LastValueQuant/AssignMaxLast�+quant_dense_58/LastValueQuant/AssignMinLast�5quant_dense_58/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_58/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_58/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_58/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_58/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_58/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_59/BiasAdd/ReadVariableOp�+quant_dense_59/LastValueQuant/AssignMaxLast�+quant_dense_59/LastValueQuant/AssignMinLast�5quant_dense_59/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_59/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_59/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_59/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_59/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_59/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_60/BiasAdd/ReadVariableOp�+quant_dense_60/LastValueQuant/AssignMaxLast�+quant_dense_60/LastValueQuant/AssignMinLast�5quant_dense_60/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_60/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_60/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_60/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_60/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_60/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_61/BiasAdd/ReadVariableOp�+quant_dense_61/LastValueQuant/AssignMaxLast�+quant_dense_61/LastValueQuant/AssignMinLast�5quant_dense_61/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_61/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_61/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_61/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_61/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_61/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_62/BiasAdd/ReadVariableOp�+quant_dense_62/LastValueQuant/AssignMaxLast�+quant_dense_62/LastValueQuant/AssignMinLast�5quant_dense_62/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_62/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_62/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_62/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_62/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_62/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_63/BiasAdd/ReadVariableOp�+quant_dense_63/LastValueQuant/AssignMaxLast�+quant_dense_63/LastValueQuant/AssignMinLast�5quant_dense_63/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_63/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_63/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_63/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_63/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_63/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�5quantize_layer_15/AllValuesQuantize/AssignMaxAllValue�5quantize_layer_15/AllValuesQuantize/AssignMinAllValue�Jquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Lquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�:quantize_layer_15/AllValuesQuantize/Maximum/ReadVariableOp�:quantize_layer_15/AllValuesQuantize/Minimum/ReadVariableOpz
)quantize_layer_15/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
,quantize_layer_15/AllValuesQuantize/BatchMinMininputs2quantize_layer_15/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: |
+quantize_layer_15/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
,quantize_layer_15/AllValuesQuantize/BatchMaxMaxinputs4quantize_layer_15/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
:quantize_layer_15/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOpCquantize_layer_15_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
+quantize_layer_15/AllValuesQuantize/MinimumMinimumBquantize_layer_15/AllValuesQuantize/Minimum/ReadVariableOp:value:05quantize_layer_15/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: t
/quantize_layer_15/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-quantize_layer_15/AllValuesQuantize/Minimum_1Minimum/quantize_layer_15/AllValuesQuantize/Minimum:z:08quantize_layer_15/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
:quantize_layer_15/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOpCquantize_layer_15_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
+quantize_layer_15/AllValuesQuantize/MaximumMaximumBquantize_layer_15/AllValuesQuantize/Maximum/ReadVariableOp:value:05quantize_layer_15/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: t
/quantize_layer_15/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-quantize_layer_15/AllValuesQuantize/Maximum_1Maximum/quantize_layer_15/AllValuesQuantize/Maximum:z:08quantize_layer_15/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
5quantize_layer_15/AllValuesQuantize/AssignMinAllValueAssignVariableOpCquantize_layer_15_allvaluesquantize_minimum_readvariableop_resource1quantize_layer_15/AllValuesQuantize/Minimum_1:z:0;^quantize_layer_15/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0�
5quantize_layer_15/AllValuesQuantize/AssignMaxAllValueAssignVariableOpCquantize_layer_15_allvaluesquantize_maximum_readvariableop_resource1quantize_layer_15/AllValuesQuantize/Maximum_1:z:0;^quantize_layer_15/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0�
Jquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpCquantize_layer_15_allvaluesquantize_minimum_readvariableop_resource6^quantize_layer_15/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
Lquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCquantize_layer_15_allvaluesquantize_maximum_readvariableop_resource6^quantize_layer_15/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
;quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsRquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Tquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
1quant_dense_53/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_53_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0d
"quant_dense_53/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_53/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_53/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_53/LastValueQuant/rangeRange2quant_dense_53/LastValueQuant/range/start:output:0+quant_dense_53/LastValueQuant/Rank:output:02quant_dense_53/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_53/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_53_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0�
&quant_dense_53/LastValueQuant/BatchMinMin=quant_dense_53/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_53/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_53/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_53_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0f
$quant_dense_53/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_53/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_53/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_53/LastValueQuant/range_1Range4quant_dense_53/LastValueQuant/range_1/start:output:0-quant_dense_53/LastValueQuant/Rank_1:output:04quant_dense_53/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_53/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_53_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0�
&quant_dense_53/LastValueQuant/BatchMaxMax=quant_dense_53/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_53/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_53/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_53/LastValueQuant/truedivRealDiv/quant_dense_53/LastValueQuant/BatchMax:output:00quant_dense_53/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_53/LastValueQuant/MinimumMinimum/quant_dense_53/LastValueQuant/BatchMin:output:0)quant_dense_53/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_53/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_53/LastValueQuant/mulMul/quant_dense_53/LastValueQuant/BatchMin:output:0,quant_dense_53/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_53/LastValueQuant/MaximumMaximum/quant_dense_53/LastValueQuant/BatchMax:output:0%quant_dense_53/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_53/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_53_lastvaluequant_assignminlast_resource)quant_dense_53/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_53/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_53_lastvaluequant_assignmaxlast_resource)quant_dense_53/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_53_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_53_lastvaluequant_assignminlast_resource,^quant_dense_53/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_53_lastvaluequant_assignmaxlast_resource,^quant_dense_53/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@@*
narrow_range(�
quant_dense_53/MatMulMatMulEquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
%quant_dense_53/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_53_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_53/BiasAddBiasAddquant_dense_53/MatMul:product:0-quant_dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@n
quant_dense_53/ReluReluquant_dense_53/BiasAdd:output:0*
T0*'
_output_shapes
:���������@w
&quant_dense_53/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_53/MovingAvgQuantize/BatchMinMin!quant_dense_53/Relu:activations:0/quant_dense_53/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_53/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_53/MovingAvgQuantize/BatchMaxMax!quant_dense_53/Relu:activations:01quant_dense_53/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_53/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_53/MovingAvgQuantize/MinimumMinimum2quant_dense_53/MovingAvgQuantize/BatchMin:output:03quant_dense_53/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_53/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_53/MovingAvgQuantize/MaximumMaximum2quant_dense_53/MovingAvgQuantize/BatchMax:output:03quant_dense_53/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_53/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_53/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_53_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_53/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_53/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_53/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_53/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_53/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_53/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_53/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_53_movingavgquantize_assignminema_readvariableop_resource5quant_dense_53/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_53/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_53/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_53/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_53_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_53/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_53/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_53/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_53/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_53/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_53/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_53/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_53_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_53/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_53/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_53_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_53/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_53_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_53/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_53/Relu:activations:0Oquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
1quant_dense_54/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_54_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@2*
dtype0d
"quant_dense_54/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_54/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_54/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_54/LastValueQuant/rangeRange2quant_dense_54/LastValueQuant/range/start:output:0+quant_dense_54/LastValueQuant/Rank:output:02quant_dense_54/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_54/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_54_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@2*
dtype0�
&quant_dense_54/LastValueQuant/BatchMinMin=quant_dense_54/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_54/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_54/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_54_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@2*
dtype0f
$quant_dense_54/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_54/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_54/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_54/LastValueQuant/range_1Range4quant_dense_54/LastValueQuant/range_1/start:output:0-quant_dense_54/LastValueQuant/Rank_1:output:04quant_dense_54/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_54/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_54_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@2*
dtype0�
&quant_dense_54/LastValueQuant/BatchMaxMax=quant_dense_54/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_54/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_54/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_54/LastValueQuant/truedivRealDiv/quant_dense_54/LastValueQuant/BatchMax:output:00quant_dense_54/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_54/LastValueQuant/MinimumMinimum/quant_dense_54/LastValueQuant/BatchMin:output:0)quant_dense_54/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_54/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_54/LastValueQuant/mulMul/quant_dense_54/LastValueQuant/BatchMin:output:0,quant_dense_54/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_54/LastValueQuant/MaximumMaximum/quant_dense_54/LastValueQuant/BatchMax:output:0%quant_dense_54/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_54/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_54_lastvaluequant_assignminlast_resource)quant_dense_54/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_54/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_54_lastvaluequant_assignmaxlast_resource)quant_dense_54/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_54_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@2*
dtype0�
Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_54_lastvaluequant_assignminlast_resource,^quant_dense_54/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_54_lastvaluequant_assignmaxlast_resource,^quant_dense_54/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@2*
narrow_range(�
quant_dense_54/MatMulMatMulBquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������2�
%quant_dense_54/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_54_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
quant_dense_54/BiasAddBiasAddquant_dense_54/MatMul:product:0-quant_dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2n
quant_dense_54/ReluReluquant_dense_54/BiasAdd:output:0*
T0*'
_output_shapes
:���������2w
&quant_dense_54/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_54/MovingAvgQuantize/BatchMinMin!quant_dense_54/Relu:activations:0/quant_dense_54/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_54/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_54/MovingAvgQuantize/BatchMaxMax!quant_dense_54/Relu:activations:01quant_dense_54/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_54/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_54/MovingAvgQuantize/MinimumMinimum2quant_dense_54/MovingAvgQuantize/BatchMin:output:03quant_dense_54/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_54/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_54/MovingAvgQuantize/MaximumMaximum2quant_dense_54/MovingAvgQuantize/BatchMax:output:03quant_dense_54/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_54/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_54/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_54_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_54/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_54/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_54/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_54/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_54/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_54/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_54/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_54_movingavgquantize_assignminema_readvariableop_resource5quant_dense_54/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_54/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_54/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_54/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_54_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_54/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_54/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_54/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_54/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_54/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_54/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_54/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_54_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_54/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_54/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_54_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_54/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_54_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_54/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_54/Relu:activations:0Oquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������2�
1quant_dense_55/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_55_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:2 *
dtype0d
"quant_dense_55/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_55/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_55/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_55/LastValueQuant/rangeRange2quant_dense_55/LastValueQuant/range/start:output:0+quant_dense_55/LastValueQuant/Rank:output:02quant_dense_55/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_55/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_55_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:2 *
dtype0�
&quant_dense_55/LastValueQuant/BatchMinMin=quant_dense_55/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_55/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_55/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_55_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:2 *
dtype0f
$quant_dense_55/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_55/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_55/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_55/LastValueQuant/range_1Range4quant_dense_55/LastValueQuant/range_1/start:output:0-quant_dense_55/LastValueQuant/Rank_1:output:04quant_dense_55/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_55/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_55_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:2 *
dtype0�
&quant_dense_55/LastValueQuant/BatchMaxMax=quant_dense_55/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_55/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_55/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_55/LastValueQuant/truedivRealDiv/quant_dense_55/LastValueQuant/BatchMax:output:00quant_dense_55/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_55/LastValueQuant/MinimumMinimum/quant_dense_55/LastValueQuant/BatchMin:output:0)quant_dense_55/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_55/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_55/LastValueQuant/mulMul/quant_dense_55/LastValueQuant/BatchMin:output:0,quant_dense_55/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_55/LastValueQuant/MaximumMaximum/quant_dense_55/LastValueQuant/BatchMax:output:0%quant_dense_55/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_55/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_55_lastvaluequant_assignminlast_resource)quant_dense_55/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_55/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_55_lastvaluequant_assignmaxlast_resource)quant_dense_55/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_55_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:2 *
dtype0�
Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_55_lastvaluequant_assignminlast_resource,^quant_dense_55/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_55_lastvaluequant_assignmaxlast_resource,^quant_dense_55/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:2 *
narrow_range(�
quant_dense_55/MatMulMatMulBquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� �
%quant_dense_55/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_55_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
quant_dense_55/BiasAddBiasAddquant_dense_55/MatMul:product:0-quant_dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
quant_dense_55/ReluReluquant_dense_55/BiasAdd:output:0*
T0*'
_output_shapes
:��������� w
&quant_dense_55/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_55/MovingAvgQuantize/BatchMinMin!quant_dense_55/Relu:activations:0/quant_dense_55/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_55/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_55/MovingAvgQuantize/BatchMaxMax!quant_dense_55/Relu:activations:01quant_dense_55/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_55/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_55/MovingAvgQuantize/MinimumMinimum2quant_dense_55/MovingAvgQuantize/BatchMin:output:03quant_dense_55/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_55/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_55/MovingAvgQuantize/MaximumMaximum2quant_dense_55/MovingAvgQuantize/BatchMax:output:03quant_dense_55/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_55/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_55/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_55_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_55/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_55/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_55/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_55/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_55/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_55/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_55/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_55_movingavgquantize_assignminema_readvariableop_resource5quant_dense_55/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_55/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_55/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_55/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_55_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_55/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_55/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_55/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_55/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_55/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_55/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_55/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_55_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_55/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_55/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_55_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_55/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_55_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_55/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_55/Relu:activations:0Oquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:��������� �
1quant_dense_56/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_56_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0d
"quant_dense_56/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_56/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_56/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_56/LastValueQuant/rangeRange2quant_dense_56/LastValueQuant/range/start:output:0+quant_dense_56/LastValueQuant/Rank:output:02quant_dense_56/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_56/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_56_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0�
&quant_dense_56/LastValueQuant/BatchMinMin=quant_dense_56/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_56/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_56/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_56_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0f
$quant_dense_56/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_56/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_56/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_56/LastValueQuant/range_1Range4quant_dense_56/LastValueQuant/range_1/start:output:0-quant_dense_56/LastValueQuant/Rank_1:output:04quant_dense_56/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_56/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_56_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0�
&quant_dense_56/LastValueQuant/BatchMaxMax=quant_dense_56/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_56/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_56/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_56/LastValueQuant/truedivRealDiv/quant_dense_56/LastValueQuant/BatchMax:output:00quant_dense_56/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_56/LastValueQuant/MinimumMinimum/quant_dense_56/LastValueQuant/BatchMin:output:0)quant_dense_56/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_56/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_56/LastValueQuant/mulMul/quant_dense_56/LastValueQuant/BatchMin:output:0,quant_dense_56/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_56/LastValueQuant/MaximumMaximum/quant_dense_56/LastValueQuant/BatchMax:output:0%quant_dense_56/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_56/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_56_lastvaluequant_assignminlast_resource)quant_dense_56/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_56/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_56_lastvaluequant_assignmaxlast_resource)quant_dense_56/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_56_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0�
Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_56_lastvaluequant_assignminlast_resource,^quant_dense_56/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_56_lastvaluequant_assignmaxlast_resource,^quant_dense_56/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

: *
narrow_range(�
quant_dense_56/MatMulMatMulBquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_56/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_56/BiasAddBiasAddquant_dense_56/MatMul:product:0-quant_dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_56/ReluReluquant_dense_56/BiasAdd:output:0*
T0*'
_output_shapes
:���������w
&quant_dense_56/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_56/MovingAvgQuantize/BatchMinMin!quant_dense_56/Relu:activations:0/quant_dense_56/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_56/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_56/MovingAvgQuantize/BatchMaxMax!quant_dense_56/Relu:activations:01quant_dense_56/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_56/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_56/MovingAvgQuantize/MinimumMinimum2quant_dense_56/MovingAvgQuantize/BatchMin:output:03quant_dense_56/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_56/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_56/MovingAvgQuantize/MaximumMaximum2quant_dense_56/MovingAvgQuantize/BatchMax:output:03quant_dense_56/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_56/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_56/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_56_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_56/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_56/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_56/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_56/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_56/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_56/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_56/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_56_movingavgquantize_assignminema_readvariableop_resource5quant_dense_56/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_56/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_56/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_56/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_56_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_56/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_56/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_56/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_56/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_56/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_56/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_56/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_56_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_56/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_56/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_56_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_56/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_56_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_56/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_56/Relu:activations:0Oquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
1quant_dense_57/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_57_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0d
"quant_dense_57/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_57/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_57/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_57/LastValueQuant/rangeRange2quant_dense_57/LastValueQuant/range/start:output:0+quant_dense_57/LastValueQuant/Rank:output:02quant_dense_57/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_57/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_57_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_57/LastValueQuant/BatchMinMin=quant_dense_57/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_57/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_57/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_57_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0f
$quant_dense_57/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_57/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_57/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_57/LastValueQuant/range_1Range4quant_dense_57/LastValueQuant/range_1/start:output:0-quant_dense_57/LastValueQuant/Rank_1:output:04quant_dense_57/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_57/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_57_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_57/LastValueQuant/BatchMaxMax=quant_dense_57/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_57/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_57/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_57/LastValueQuant/truedivRealDiv/quant_dense_57/LastValueQuant/BatchMax:output:00quant_dense_57/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_57/LastValueQuant/MinimumMinimum/quant_dense_57/LastValueQuant/BatchMin:output:0)quant_dense_57/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_57/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_57/LastValueQuant/mulMul/quant_dense_57/LastValueQuant/BatchMin:output:0,quant_dense_57/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_57/LastValueQuant/MaximumMaximum/quant_dense_57/LastValueQuant/BatchMax:output:0%quant_dense_57/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_57/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_57_lastvaluequant_assignminlast_resource)quant_dense_57/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_57/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_57_lastvaluequant_assignmaxlast_resource)quant_dense_57/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_57_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_57_lastvaluequant_assignminlast_resource,^quant_dense_57/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_57_lastvaluequant_assignmaxlast_resource,^quant_dense_57/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_57/MatMulMatMulBquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_57/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_57/BiasAddBiasAddquant_dense_57/MatMul:product:0-quant_dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_57/ReluReluquant_dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:���������w
&quant_dense_57/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_57/MovingAvgQuantize/BatchMinMin!quant_dense_57/Relu:activations:0/quant_dense_57/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_57/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_57/MovingAvgQuantize/BatchMaxMax!quant_dense_57/Relu:activations:01quant_dense_57/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_57/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_57/MovingAvgQuantize/MinimumMinimum2quant_dense_57/MovingAvgQuantize/BatchMin:output:03quant_dense_57/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_57/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_57/MovingAvgQuantize/MaximumMaximum2quant_dense_57/MovingAvgQuantize/BatchMax:output:03quant_dense_57/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_57/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_57/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_57_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_57/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_57/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_57/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_57/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_57/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_57/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_57/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_57_movingavgquantize_assignminema_readvariableop_resource5quant_dense_57/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_57/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_57/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_57/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_57_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_57/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_57/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_57/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_57/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_57/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_57/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_57/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_57_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_57/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_57/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_57_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_57/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_57_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_57/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_57/Relu:activations:0Oquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
1quant_dense_58/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_58_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0d
"quant_dense_58/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_58/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_58/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_58/LastValueQuant/rangeRange2quant_dense_58/LastValueQuant/range/start:output:0+quant_dense_58/LastValueQuant/Rank:output:02quant_dense_58/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_58/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_58_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_58/LastValueQuant/BatchMinMin=quant_dense_58/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_58/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_58/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_58_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0f
$quant_dense_58/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_58/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_58/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_58/LastValueQuant/range_1Range4quant_dense_58/LastValueQuant/range_1/start:output:0-quant_dense_58/LastValueQuant/Rank_1:output:04quant_dense_58/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_58/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_58_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_58/LastValueQuant/BatchMaxMax=quant_dense_58/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_58/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_58/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_58/LastValueQuant/truedivRealDiv/quant_dense_58/LastValueQuant/BatchMax:output:00quant_dense_58/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_58/LastValueQuant/MinimumMinimum/quant_dense_58/LastValueQuant/BatchMin:output:0)quant_dense_58/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_58/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_58/LastValueQuant/mulMul/quant_dense_58/LastValueQuant/BatchMin:output:0,quant_dense_58/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_58/LastValueQuant/MaximumMaximum/quant_dense_58/LastValueQuant/BatchMax:output:0%quant_dense_58/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_58/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_58_lastvaluequant_assignminlast_resource)quant_dense_58/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_58/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_58_lastvaluequant_assignmaxlast_resource)quant_dense_58/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_58_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_58_lastvaluequant_assignminlast_resource,^quant_dense_58/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_58_lastvaluequant_assignmaxlast_resource,^quant_dense_58/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_58/MatMulMatMulBquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_58/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_58/BiasAddBiasAddquant_dense_58/MatMul:product:0-quant_dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_58/ReluReluquant_dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:���������w
&quant_dense_58/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_58/MovingAvgQuantize/BatchMinMin!quant_dense_58/Relu:activations:0/quant_dense_58/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_58/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_58/MovingAvgQuantize/BatchMaxMax!quant_dense_58/Relu:activations:01quant_dense_58/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_58/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_58/MovingAvgQuantize/MinimumMinimum2quant_dense_58/MovingAvgQuantize/BatchMin:output:03quant_dense_58/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_58/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_58/MovingAvgQuantize/MaximumMaximum2quant_dense_58/MovingAvgQuantize/BatchMax:output:03quant_dense_58/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_58/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_58/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_58_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_58/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_58/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_58/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_58/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_58/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_58/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_58/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_58_movingavgquantize_assignminema_readvariableop_resource5quant_dense_58/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_58/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_58/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_58/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_58_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_58/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_58/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_58/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_58/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_58/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_58/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_58/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_58_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_58/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_58/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_58_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_58/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_58_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_58/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_58/Relu:activations:0Oquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
1quant_dense_59/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_59_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0d
"quant_dense_59/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_59/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_59/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_59/LastValueQuant/rangeRange2quant_dense_59/LastValueQuant/range/start:output:0+quant_dense_59/LastValueQuant/Rank:output:02quant_dense_59/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_59/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_59_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_59/LastValueQuant/BatchMinMin=quant_dense_59/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_59/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_59/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_59_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0f
$quant_dense_59/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_59/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_59/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_59/LastValueQuant/range_1Range4quant_dense_59/LastValueQuant/range_1/start:output:0-quant_dense_59/LastValueQuant/Rank_1:output:04quant_dense_59/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_59/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_59_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_59/LastValueQuant/BatchMaxMax=quant_dense_59/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_59/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_59/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_59/LastValueQuant/truedivRealDiv/quant_dense_59/LastValueQuant/BatchMax:output:00quant_dense_59/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_59/LastValueQuant/MinimumMinimum/quant_dense_59/LastValueQuant/BatchMin:output:0)quant_dense_59/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_59/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_59/LastValueQuant/mulMul/quant_dense_59/LastValueQuant/BatchMin:output:0,quant_dense_59/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_59/LastValueQuant/MaximumMaximum/quant_dense_59/LastValueQuant/BatchMax:output:0%quant_dense_59/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_59/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_59_lastvaluequant_assignminlast_resource)quant_dense_59/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_59/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_59_lastvaluequant_assignmaxlast_resource)quant_dense_59/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_59_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_59_lastvaluequant_assignminlast_resource,^quant_dense_59/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_59_lastvaluequant_assignmaxlast_resource,^quant_dense_59/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_59/MatMulMatMulBquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_59/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_59/BiasAddBiasAddquant_dense_59/MatMul:product:0-quant_dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_59/ReluReluquant_dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:���������w
&quant_dense_59/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_59/MovingAvgQuantize/BatchMinMin!quant_dense_59/Relu:activations:0/quant_dense_59/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_59/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_59/MovingAvgQuantize/BatchMaxMax!quant_dense_59/Relu:activations:01quant_dense_59/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_59/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_59/MovingAvgQuantize/MinimumMinimum2quant_dense_59/MovingAvgQuantize/BatchMin:output:03quant_dense_59/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_59/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_59/MovingAvgQuantize/MaximumMaximum2quant_dense_59/MovingAvgQuantize/BatchMax:output:03quant_dense_59/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_59/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_59/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_59_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_59/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_59/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_59/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_59/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_59/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_59/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_59/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_59_movingavgquantize_assignminema_readvariableop_resource5quant_dense_59/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_59/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_59/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_59/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_59_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_59/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_59/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_59/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_59/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_59/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_59/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_59/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_59_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_59/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_59/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_59_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_59/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_59_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_59/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_59/Relu:activations:0Oquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
1quant_dense_60/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_60_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0d
"quant_dense_60/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_60/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_60/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_60/LastValueQuant/rangeRange2quant_dense_60/LastValueQuant/range/start:output:0+quant_dense_60/LastValueQuant/Rank:output:02quant_dense_60/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_60/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_60_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_60/LastValueQuant/BatchMinMin=quant_dense_60/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_60/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_60/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_60_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0f
$quant_dense_60/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_60/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_60/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_60/LastValueQuant/range_1Range4quant_dense_60/LastValueQuant/range_1/start:output:0-quant_dense_60/LastValueQuant/Rank_1:output:04quant_dense_60/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_60/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_60_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
&quant_dense_60/LastValueQuant/BatchMaxMax=quant_dense_60/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_60/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_60/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_60/LastValueQuant/truedivRealDiv/quant_dense_60/LastValueQuant/BatchMax:output:00quant_dense_60/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_60/LastValueQuant/MinimumMinimum/quant_dense_60/LastValueQuant/BatchMin:output:0)quant_dense_60/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_60/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_60/LastValueQuant/mulMul/quant_dense_60/LastValueQuant/BatchMin:output:0,quant_dense_60/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_60/LastValueQuant/MaximumMaximum/quant_dense_60/LastValueQuant/BatchMax:output:0%quant_dense_60/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_60/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_60_lastvaluequant_assignminlast_resource)quant_dense_60/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_60/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_60_lastvaluequant_assignmaxlast_resource)quant_dense_60/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_60_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_60_lastvaluequant_assignminlast_resource,^quant_dense_60/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_60_lastvaluequant_assignmaxlast_resource,^quant_dense_60/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_60/MatMulMatMulBquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_60/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_60/BiasAddBiasAddquant_dense_60/MatMul:product:0-quant_dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_60/ReluReluquant_dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:���������w
&quant_dense_60/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_60/MovingAvgQuantize/BatchMinMin!quant_dense_60/Relu:activations:0/quant_dense_60/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_60/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_60/MovingAvgQuantize/BatchMaxMax!quant_dense_60/Relu:activations:01quant_dense_60/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_60/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_60/MovingAvgQuantize/MinimumMinimum2quant_dense_60/MovingAvgQuantize/BatchMin:output:03quant_dense_60/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_60/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_60/MovingAvgQuantize/MaximumMaximum2quant_dense_60/MovingAvgQuantize/BatchMax:output:03quant_dense_60/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_60/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_60/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_60_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_60/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_60/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_60/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_60/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_60/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_60/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_60/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_60_movingavgquantize_assignminema_readvariableop_resource5quant_dense_60/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_60/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_60/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_60/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_60_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_60/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_60/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_60/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_60/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_60/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_60/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_60/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_60_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_60/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_60/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_60_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_60/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_60_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_60/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_60/Relu:activations:0Oquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
1quant_dense_61/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_61_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0d
"quant_dense_61/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_61/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_61/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_61/LastValueQuant/rangeRange2quant_dense_61/LastValueQuant/range/start:output:0+quant_dense_61/LastValueQuant/Rank:output:02quant_dense_61/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_61/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_61_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0�
&quant_dense_61/LastValueQuant/BatchMinMin=quant_dense_61/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_61/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_61/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_61_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0f
$quant_dense_61/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_61/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_61/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_61/LastValueQuant/range_1Range4quant_dense_61/LastValueQuant/range_1/start:output:0-quant_dense_61/LastValueQuant/Rank_1:output:04quant_dense_61/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_61/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_61_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0�
&quant_dense_61/LastValueQuant/BatchMaxMax=quant_dense_61/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_61/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_61/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_61/LastValueQuant/truedivRealDiv/quant_dense_61/LastValueQuant/BatchMax:output:00quant_dense_61/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_61/LastValueQuant/MinimumMinimum/quant_dense_61/LastValueQuant/BatchMin:output:0)quant_dense_61/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_61/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_61/LastValueQuant/mulMul/quant_dense_61/LastValueQuant/BatchMin:output:0,quant_dense_61/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_61/LastValueQuant/MaximumMaximum/quant_dense_61/LastValueQuant/BatchMax:output:0%quant_dense_61/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_61/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_61_lastvaluequant_assignminlast_resource)quant_dense_61/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_61/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_61_lastvaluequant_assignmaxlast_resource)quant_dense_61/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_61_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
dtype0�
Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_61_lastvaluequant_assignminlast_resource,^quant_dense_61/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_61_lastvaluequant_assignmaxlast_resource,^quant_dense_61/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

: *
narrow_range(�
quant_dense_61/MatMulMatMulBquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� �
%quant_dense_61/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
quant_dense_61/BiasAddBiasAddquant_dense_61/MatMul:product:0-quant_dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
quant_dense_61/ReluReluquant_dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� w
&quant_dense_61/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_61/MovingAvgQuantize/BatchMinMin!quant_dense_61/Relu:activations:0/quant_dense_61/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_61/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_61/MovingAvgQuantize/BatchMaxMax!quant_dense_61/Relu:activations:01quant_dense_61/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_61/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_61/MovingAvgQuantize/MinimumMinimum2quant_dense_61/MovingAvgQuantize/BatchMin:output:03quant_dense_61/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_61/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_61/MovingAvgQuantize/MaximumMaximum2quant_dense_61/MovingAvgQuantize/BatchMax:output:03quant_dense_61/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_61/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_61/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_61_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_61/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_61/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_61/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_61/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_61/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_61/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_61/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_61_movingavgquantize_assignminema_readvariableop_resource5quant_dense_61/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_61/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_61/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_61/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_61_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_61/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_61/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_61/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_61/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_61/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_61/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_61/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_61_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_61/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_61/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_61_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_61/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_61_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_61/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_61/Relu:activations:0Oquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:��������� �
1quant_dense_62/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_62_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:  *
dtype0d
"quant_dense_62/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_62/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_62/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_62/LastValueQuant/rangeRange2quant_dense_62/LastValueQuant/range/start:output:0+quant_dense_62/LastValueQuant/Rank:output:02quant_dense_62/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_62/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_62_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:  *
dtype0�
&quant_dense_62/LastValueQuant/BatchMinMin=quant_dense_62/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_62/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_62/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_62_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:  *
dtype0f
$quant_dense_62/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_62/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_62/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_62/LastValueQuant/range_1Range4quant_dense_62/LastValueQuant/range_1/start:output:0-quant_dense_62/LastValueQuant/Rank_1:output:04quant_dense_62/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_62/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_62_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:  *
dtype0�
&quant_dense_62/LastValueQuant/BatchMaxMax=quant_dense_62/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_62/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_62/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_62/LastValueQuant/truedivRealDiv/quant_dense_62/LastValueQuant/BatchMax:output:00quant_dense_62/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_62/LastValueQuant/MinimumMinimum/quant_dense_62/LastValueQuant/BatchMin:output:0)quant_dense_62/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_62/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_62/LastValueQuant/mulMul/quant_dense_62/LastValueQuant/BatchMin:output:0,quant_dense_62/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_62/LastValueQuant/MaximumMaximum/quant_dense_62/LastValueQuant/BatchMax:output:0%quant_dense_62/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_62/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_62_lastvaluequant_assignminlast_resource)quant_dense_62/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_62/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_62_lastvaluequant_assignmaxlast_resource)quant_dense_62/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_62_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:  *
dtype0�
Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_62_lastvaluequant_assignminlast_resource,^quant_dense_62/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_62_lastvaluequant_assignmaxlast_resource,^quant_dense_62/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:  *
narrow_range(�
quant_dense_62/MatMulMatMulBquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� �
%quant_dense_62/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_62_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
quant_dense_62/BiasAddBiasAddquant_dense_62/MatMul:product:0-quant_dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
quant_dense_62/ReluReluquant_dense_62/BiasAdd:output:0*
T0*'
_output_shapes
:��������� w
&quant_dense_62/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_62/MovingAvgQuantize/BatchMinMin!quant_dense_62/Relu:activations:0/quant_dense_62/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_62/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_62/MovingAvgQuantize/BatchMaxMax!quant_dense_62/Relu:activations:01quant_dense_62/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_62/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_62/MovingAvgQuantize/MinimumMinimum2quant_dense_62/MovingAvgQuantize/BatchMin:output:03quant_dense_62/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_62/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_62/MovingAvgQuantize/MaximumMaximum2quant_dense_62/MovingAvgQuantize/BatchMax:output:03quant_dense_62/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_62/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_62/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_62_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_62/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_62/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_62/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_62/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_62/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_62/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_62/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_62_movingavgquantize_assignminema_readvariableop_resource5quant_dense_62/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_62/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_62/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_62/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_62_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_62/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_62/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_62/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_62/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_62/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_62/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_62/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_62_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_62/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_62/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_62_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_62/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_62_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_62/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_62/Relu:activations:0Oquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:��������� �
1quant_dense_63/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_63_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: @*
dtype0d
"quant_dense_63/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_63/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_63/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_63/LastValueQuant/rangeRange2quant_dense_63/LastValueQuant/range/start:output:0+quant_dense_63/LastValueQuant/Rank:output:02quant_dense_63/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_63/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_63_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: @*
dtype0�
&quant_dense_63/LastValueQuant/BatchMinMin=quant_dense_63/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_63/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_63/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_63_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: @*
dtype0f
$quant_dense_63/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_63/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_63/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_63/LastValueQuant/range_1Range4quant_dense_63/LastValueQuant/range_1/start:output:0-quant_dense_63/LastValueQuant/Rank_1:output:04quant_dense_63/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_63/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_63_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: @*
dtype0�
&quant_dense_63/LastValueQuant/BatchMaxMax=quant_dense_63/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_63/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_63/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_63/LastValueQuant/truedivRealDiv/quant_dense_63/LastValueQuant/BatchMax:output:00quant_dense_63/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_63/LastValueQuant/MinimumMinimum/quant_dense_63/LastValueQuant/BatchMin:output:0)quant_dense_63/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_63/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_63/LastValueQuant/mulMul/quant_dense_63/LastValueQuant/BatchMin:output:0,quant_dense_63/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_63/LastValueQuant/MaximumMaximum/quant_dense_63/LastValueQuant/BatchMax:output:0%quant_dense_63/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_63/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_63_lastvaluequant_assignminlast_resource)quant_dense_63/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_63/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_63_lastvaluequant_assignmaxlast_resource)quant_dense_63/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_63_lastvaluequant_rank_readvariableop_resource*
_output_shapes

: @*
dtype0�
Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_63_lastvaluequant_assignminlast_resource,^quant_dense_63/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_63_lastvaluequant_assignmaxlast_resource,^quant_dense_63/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

: @*
narrow_range(�
quant_dense_63/MatMulMatMulBquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
%quant_dense_63/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_63_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_63/BiasAddBiasAddquant_dense_63/MatMul:product:0-quant_dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@w
&quant_dense_63/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_63/MovingAvgQuantize/BatchMinMinquant_dense_63/BiasAdd:output:0/quant_dense_63/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_63/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_63/MovingAvgQuantize/BatchMaxMaxquant_dense_63/BiasAdd:output:01quant_dense_63/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_63/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_63/MovingAvgQuantize/MinimumMinimum2quant_dense_63/MovingAvgQuantize/BatchMin:output:03quant_dense_63/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_63/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_63/MovingAvgQuantize/MaximumMaximum2quant_dense_63/MovingAvgQuantize/BatchMax:output:03quant_dense_63/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_63/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_63/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_63_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_63/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_63/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_63/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_63/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_63/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_63/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_63/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_63_movingavgquantize_assignminema_readvariableop_resource5quant_dense_63/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_63/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_63/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_63/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_63_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_63/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_63/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_63/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_63/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_63/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_63/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_63/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_63_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_63/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_63/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_63_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_63/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_63_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_63/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_63/BiasAdd:output:0Oquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityBquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�N
NoOpNoOp&^quant_dense_53/BiasAdd/ReadVariableOp,^quant_dense_53/LastValueQuant/AssignMaxLast,^quant_dense_53/LastValueQuant/AssignMinLast6^quant_dense_53/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_53/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_53/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_53/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_53/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_53/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_54/BiasAdd/ReadVariableOp,^quant_dense_54/LastValueQuant/AssignMaxLast,^quant_dense_54/LastValueQuant/AssignMinLast6^quant_dense_54/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_54/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_54/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_54/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_54/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_54/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_55/BiasAdd/ReadVariableOp,^quant_dense_55/LastValueQuant/AssignMaxLast,^quant_dense_55/LastValueQuant/AssignMinLast6^quant_dense_55/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_55/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_55/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_55/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_55/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_55/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_56/BiasAdd/ReadVariableOp,^quant_dense_56/LastValueQuant/AssignMaxLast,^quant_dense_56/LastValueQuant/AssignMinLast6^quant_dense_56/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_56/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_56/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_56/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_56/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_56/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_57/BiasAdd/ReadVariableOp,^quant_dense_57/LastValueQuant/AssignMaxLast,^quant_dense_57/LastValueQuant/AssignMinLast6^quant_dense_57/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_57/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_57/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_57/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_57/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_57/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_58/BiasAdd/ReadVariableOp,^quant_dense_58/LastValueQuant/AssignMaxLast,^quant_dense_58/LastValueQuant/AssignMinLast6^quant_dense_58/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_58/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_58/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_58/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_58/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_58/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_59/BiasAdd/ReadVariableOp,^quant_dense_59/LastValueQuant/AssignMaxLast,^quant_dense_59/LastValueQuant/AssignMinLast6^quant_dense_59/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_59/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_59/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_59/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_59/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_59/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_60/BiasAdd/ReadVariableOp,^quant_dense_60/LastValueQuant/AssignMaxLast,^quant_dense_60/LastValueQuant/AssignMinLast6^quant_dense_60/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_60/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_60/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_60/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_60/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_60/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_61/BiasAdd/ReadVariableOp,^quant_dense_61/LastValueQuant/AssignMaxLast,^quant_dense_61/LastValueQuant/AssignMinLast6^quant_dense_61/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_61/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_61/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_61/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_61/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_61/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_62/BiasAdd/ReadVariableOp,^quant_dense_62/LastValueQuant/AssignMaxLast,^quant_dense_62/LastValueQuant/AssignMinLast6^quant_dense_62/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_62/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_62/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_62/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_62/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_62/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_63/BiasAdd/ReadVariableOp,^quant_dense_63/LastValueQuant/AssignMaxLast,^quant_dense_63/LastValueQuant/AssignMinLast6^quant_dense_63/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_63/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_63/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_63/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_63/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_63/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_16^quantize_layer_15/AllValuesQuantize/AssignMaxAllValue6^quantize_layer_15/AllValuesQuantize/AssignMinAllValueK^quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpM^quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1;^quantize_layer_15/AllValuesQuantize/Maximum/ReadVariableOp;^quantize_layer_15/AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%quant_dense_53/BiasAdd/ReadVariableOp%quant_dense_53/BiasAdd/ReadVariableOp2Z
+quant_dense_53/LastValueQuant/AssignMaxLast+quant_dense_53/LastValueQuant/AssignMaxLast2Z
+quant_dense_53/LastValueQuant/AssignMinLast+quant_dense_53/LastValueQuant/AssignMinLast2n
5quant_dense_53/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_53/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_53/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_53/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_53/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_53/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_53/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_53/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_53/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_53/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_53/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_53/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_54/BiasAdd/ReadVariableOp%quant_dense_54/BiasAdd/ReadVariableOp2Z
+quant_dense_54/LastValueQuant/AssignMaxLast+quant_dense_54/LastValueQuant/AssignMaxLast2Z
+quant_dense_54/LastValueQuant/AssignMinLast+quant_dense_54/LastValueQuant/AssignMinLast2n
5quant_dense_54/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_54/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_54/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_54/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_54/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_54/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_54/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_54/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_54/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_54/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_54/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_54/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_55/BiasAdd/ReadVariableOp%quant_dense_55/BiasAdd/ReadVariableOp2Z
+quant_dense_55/LastValueQuant/AssignMaxLast+quant_dense_55/LastValueQuant/AssignMaxLast2Z
+quant_dense_55/LastValueQuant/AssignMinLast+quant_dense_55/LastValueQuant/AssignMinLast2n
5quant_dense_55/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_55/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_55/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_55/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_55/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_55/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_55/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_55/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_55/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_55/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_55/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_55/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_56/BiasAdd/ReadVariableOp%quant_dense_56/BiasAdd/ReadVariableOp2Z
+quant_dense_56/LastValueQuant/AssignMaxLast+quant_dense_56/LastValueQuant/AssignMaxLast2Z
+quant_dense_56/LastValueQuant/AssignMinLast+quant_dense_56/LastValueQuant/AssignMinLast2n
5quant_dense_56/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_56/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_56/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_56/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_56/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_56/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_56/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_56/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_56/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_56/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_56/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_56/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_57/BiasAdd/ReadVariableOp%quant_dense_57/BiasAdd/ReadVariableOp2Z
+quant_dense_57/LastValueQuant/AssignMaxLast+quant_dense_57/LastValueQuant/AssignMaxLast2Z
+quant_dense_57/LastValueQuant/AssignMinLast+quant_dense_57/LastValueQuant/AssignMinLast2n
5quant_dense_57/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_57/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_57/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_57/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_57/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_57/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_57/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_57/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_57/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_57/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_57/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_57/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_58/BiasAdd/ReadVariableOp%quant_dense_58/BiasAdd/ReadVariableOp2Z
+quant_dense_58/LastValueQuant/AssignMaxLast+quant_dense_58/LastValueQuant/AssignMaxLast2Z
+quant_dense_58/LastValueQuant/AssignMinLast+quant_dense_58/LastValueQuant/AssignMinLast2n
5quant_dense_58/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_58/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_58/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_58/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_58/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_58/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_58/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_58/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_58/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_58/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_58/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_58/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_59/BiasAdd/ReadVariableOp%quant_dense_59/BiasAdd/ReadVariableOp2Z
+quant_dense_59/LastValueQuant/AssignMaxLast+quant_dense_59/LastValueQuant/AssignMaxLast2Z
+quant_dense_59/LastValueQuant/AssignMinLast+quant_dense_59/LastValueQuant/AssignMinLast2n
5quant_dense_59/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_59/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_59/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_59/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_59/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_59/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_59/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_59/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_59/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_59/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_59/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_59/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_60/BiasAdd/ReadVariableOp%quant_dense_60/BiasAdd/ReadVariableOp2Z
+quant_dense_60/LastValueQuant/AssignMaxLast+quant_dense_60/LastValueQuant/AssignMaxLast2Z
+quant_dense_60/LastValueQuant/AssignMinLast+quant_dense_60/LastValueQuant/AssignMinLast2n
5quant_dense_60/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_60/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_60/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_60/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_60/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_60/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_60/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_60/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_60/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_60/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_60/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_60/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_61/BiasAdd/ReadVariableOp%quant_dense_61/BiasAdd/ReadVariableOp2Z
+quant_dense_61/LastValueQuant/AssignMaxLast+quant_dense_61/LastValueQuant/AssignMaxLast2Z
+quant_dense_61/LastValueQuant/AssignMinLast+quant_dense_61/LastValueQuant/AssignMinLast2n
5quant_dense_61/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_61/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_61/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_61/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_61/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_61/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_61/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_61/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_61/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_61/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_61/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_61/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_62/BiasAdd/ReadVariableOp%quant_dense_62/BiasAdd/ReadVariableOp2Z
+quant_dense_62/LastValueQuant/AssignMaxLast+quant_dense_62/LastValueQuant/AssignMaxLast2Z
+quant_dense_62/LastValueQuant/AssignMinLast+quant_dense_62/LastValueQuant/AssignMinLast2n
5quant_dense_62/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_62/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_62/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_62/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_62/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_62/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_62/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_62/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_62/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_62/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_62/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_62/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_63/BiasAdd/ReadVariableOp%quant_dense_63/BiasAdd/ReadVariableOp2Z
+quant_dense_63/LastValueQuant/AssignMaxLast+quant_dense_63/LastValueQuant/AssignMaxLast2Z
+quant_dense_63/LastValueQuant/AssignMinLast+quant_dense_63/LastValueQuant/AssignMinLast2n
5quant_dense_63/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_63/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_63/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_63/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_63/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_63/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_63/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_63/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_63/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_63/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_63/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_63/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12n
5quantize_layer_15/AllValuesQuantize/AssignMaxAllValue5quantize_layer_15/AllValuesQuantize/AssignMaxAllValue2n
5quantize_layer_15/AllValuesQuantize/AssignMinAllValue5quantize_layer_15/AllValuesQuantize/AssignMinAllValue2�
Jquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Lquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Lquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12x
:quantize_layer_15/AllValuesQuantize/Maximum/ReadVariableOp:quantize_layer_15/AllValuesQuantize/Maximum/ReadVariableOp2x
:quantize_layer_15/AllValuesQuantize/Minimum/ReadVariableOp:quantize_layer_15/AllValuesQuantize/Minimum/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3750680

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: @J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: @*
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

: @*
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
:��������� : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_62_layer_call_fn_3755083

inputs
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
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
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3751002o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3750611

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: *
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

: *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Y
�
E__inference_model_15_layer_call_and_return_conditional_losses_3752180

inputs#
quantize_layer_15_3752031: #
quantize_layer_15_3752033: (
quant_dense_53_3752036:@@ 
quant_dense_53_3752038:  
quant_dense_53_3752040: $
quant_dense_53_3752042:@ 
quant_dense_53_3752044:  
quant_dense_53_3752046: (
quant_dense_54_3752049:@2 
quant_dense_54_3752051:  
quant_dense_54_3752053: $
quant_dense_54_3752055:2 
quant_dense_54_3752057:  
quant_dense_54_3752059: (
quant_dense_55_3752062:2  
quant_dense_55_3752064:  
quant_dense_55_3752066: $
quant_dense_55_3752068:  
quant_dense_55_3752070:  
quant_dense_55_3752072: (
quant_dense_56_3752075:  
quant_dense_56_3752077:  
quant_dense_56_3752079: $
quant_dense_56_3752081: 
quant_dense_56_3752083:  
quant_dense_56_3752085: (
quant_dense_57_3752088: 
quant_dense_57_3752090:  
quant_dense_57_3752092: $
quant_dense_57_3752094: 
quant_dense_57_3752096:  
quant_dense_57_3752098: (
quant_dense_58_3752101: 
quant_dense_58_3752103:  
quant_dense_58_3752105: $
quant_dense_58_3752107: 
quant_dense_58_3752109:  
quant_dense_58_3752111: (
quant_dense_59_3752114: 
quant_dense_59_3752116:  
quant_dense_59_3752118: $
quant_dense_59_3752120: 
quant_dense_59_3752122:  
quant_dense_59_3752124: (
quant_dense_60_3752127: 
quant_dense_60_3752129:  
quant_dense_60_3752131: $
quant_dense_60_3752133: 
quant_dense_60_3752135:  
quant_dense_60_3752137: (
quant_dense_61_3752140:  
quant_dense_61_3752142:  
quant_dense_61_3752144: $
quant_dense_61_3752146:  
quant_dense_61_3752148:  
quant_dense_61_3752150: (
quant_dense_62_3752153:   
quant_dense_62_3752155:  
quant_dense_62_3752157: $
quant_dense_62_3752159:  
quant_dense_62_3752161:  
quant_dense_62_3752163: (
quant_dense_63_3752166: @ 
quant_dense_63_3752168:  
quant_dense_63_3752170: $
quant_dense_63_3752172:@ 
quant_dense_63_3752174:  
quant_dense_63_3752176: 
identity��&quant_dense_53/StatefulPartitionedCall�&quant_dense_54/StatefulPartitionedCall�&quant_dense_55/StatefulPartitionedCall�&quant_dense_56/StatefulPartitionedCall�&quant_dense_57/StatefulPartitionedCall�&quant_dense_58/StatefulPartitionedCall�&quant_dense_59/StatefulPartitionedCall�&quant_dense_60/StatefulPartitionedCall�&quant_dense_61/StatefulPartitionedCall�&quant_dense_62/StatefulPartitionedCall�&quant_dense_63/StatefulPartitionedCall�)quantize_layer_15/StatefulPartitionedCall�
)quantize_layer_15/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_15_3752031quantize_layer_15_3752033*
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
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3751878�
&quant_dense_53/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_15/StatefulPartitionedCall:output:0quant_dense_53_3752036quant_dense_53_3752038quant_dense_53_3752040quant_dense_53_3752042quant_dense_53_3752044quant_dense_53_3752046*
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
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3751830�
&quant_dense_54/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_53/StatefulPartitionedCall:output:0quant_dense_54_3752049quant_dense_54_3752051quant_dense_54_3752053quant_dense_54_3752055quant_dense_54_3752057quant_dense_54_3752059*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3751738�
&quant_dense_55/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_54/StatefulPartitionedCall:output:0quant_dense_55_3752062quant_dense_55_3752064quant_dense_55_3752066quant_dense_55_3752068quant_dense_55_3752070quant_dense_55_3752072*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3751646�
&quant_dense_56/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_55/StatefulPartitionedCall:output:0quant_dense_56_3752075quant_dense_56_3752077quant_dense_56_3752079quant_dense_56_3752081quant_dense_56_3752083quant_dense_56_3752085*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3751554�
&quant_dense_57/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_56/StatefulPartitionedCall:output:0quant_dense_57_3752088quant_dense_57_3752090quant_dense_57_3752092quant_dense_57_3752094quant_dense_57_3752096quant_dense_57_3752098*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3751462�
&quant_dense_58/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_57/StatefulPartitionedCall:output:0quant_dense_58_3752101quant_dense_58_3752103quant_dense_58_3752105quant_dense_58_3752107quant_dense_58_3752109quant_dense_58_3752111*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3751370�
&quant_dense_59/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_58/StatefulPartitionedCall:output:0quant_dense_59_3752114quant_dense_59_3752116quant_dense_59_3752118quant_dense_59_3752120quant_dense_59_3752122quant_dense_59_3752124*
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
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3751278�
&quant_dense_60/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_59/StatefulPartitionedCall:output:0quant_dense_60_3752127quant_dense_60_3752129quant_dense_60_3752131quant_dense_60_3752133quant_dense_60_3752135quant_dense_60_3752137*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3751186�
&quant_dense_61/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_60/StatefulPartitionedCall:output:0quant_dense_61_3752140quant_dense_61_3752142quant_dense_61_3752144quant_dense_61_3752146quant_dense_61_3752148quant_dense_61_3752150*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3751094�
&quant_dense_62/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_61/StatefulPartitionedCall:output:0quant_dense_62_3752153quant_dense_62_3752155quant_dense_62_3752157quant_dense_62_3752159quant_dense_62_3752161quant_dense_62_3752163*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3751002�
&quant_dense_63/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_62/StatefulPartitionedCall:output:0quant_dense_63_3752166quant_dense_63_3752168quant_dense_63_3752170quant_dense_63_3752172quant_dense_63_3752174quant_dense_63_3752176*
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
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3750910~
IdentityIdentity/quant_dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_53/StatefulPartitionedCall'^quant_dense_54/StatefulPartitionedCall'^quant_dense_55/StatefulPartitionedCall'^quant_dense_56/StatefulPartitionedCall'^quant_dense_57/StatefulPartitionedCall'^quant_dense_58/StatefulPartitionedCall'^quant_dense_59/StatefulPartitionedCall'^quant_dense_60/StatefulPartitionedCall'^quant_dense_61/StatefulPartitionedCall'^quant_dense_62/StatefulPartitionedCall'^quant_dense_63/StatefulPartitionedCall*^quantize_layer_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&quant_dense_53/StatefulPartitionedCall&quant_dense_53/StatefulPartitionedCall2P
&quant_dense_54/StatefulPartitionedCall&quant_dense_54/StatefulPartitionedCall2P
&quant_dense_55/StatefulPartitionedCall&quant_dense_55/StatefulPartitionedCall2P
&quant_dense_56/StatefulPartitionedCall&quant_dense_56/StatefulPartitionedCall2P
&quant_dense_57/StatefulPartitionedCall&quant_dense_57/StatefulPartitionedCall2P
&quant_dense_58/StatefulPartitionedCall&quant_dense_58/StatefulPartitionedCall2P
&quant_dense_59/StatefulPartitionedCall&quant_dense_59/StatefulPartitionedCall2P
&quant_dense_60/StatefulPartitionedCall&quant_dense_60/StatefulPartitionedCall2P
&quant_dense_61/StatefulPartitionedCall&quant_dense_61/StatefulPartitionedCall2P
&quant_dense_62/StatefulPartitionedCall&quant_dense_62/StatefulPartitionedCall2P
&quant_dense_63/StatefulPartitionedCall&quant_dense_63/StatefulPartitionedCall2V
)quantize_layer_15/StatefulPartitionedCall)quantize_layer_15/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3754320

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:2 J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:2 *
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

:2 *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_61_layer_call_fn_3754954

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
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
:��������� *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3750611o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_54_layer_call_fn_3754170

inputs
unknown:@2
	unknown_0: 
	unknown_1: 
	unknown_2:2
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
:���������2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3750366o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
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
��
�:
 __inference__traced_save_3755687
file_prefixF
Bsavev2_quantize_layer_15_quantize_layer_15_min_read_readvariableopF
Bsavev2_quantize_layer_15_quantize_layer_15_max_read_readvariableop?
;savev2_quantize_layer_15_optimizer_step_read_readvariableop<
8savev2_quant_dense_53_optimizer_step_read_readvariableop8
4savev2_quant_dense_53_kernel_min_read_readvariableop8
4savev2_quant_dense_53_kernel_max_read_readvariableopA
=savev2_quant_dense_53_post_activation_min_read_readvariableopA
=savev2_quant_dense_53_post_activation_max_read_readvariableop<
8savev2_quant_dense_54_optimizer_step_read_readvariableop8
4savev2_quant_dense_54_kernel_min_read_readvariableop8
4savev2_quant_dense_54_kernel_max_read_readvariableopA
=savev2_quant_dense_54_post_activation_min_read_readvariableopA
=savev2_quant_dense_54_post_activation_max_read_readvariableop<
8savev2_quant_dense_55_optimizer_step_read_readvariableop8
4savev2_quant_dense_55_kernel_min_read_readvariableop8
4savev2_quant_dense_55_kernel_max_read_readvariableopA
=savev2_quant_dense_55_post_activation_min_read_readvariableopA
=savev2_quant_dense_55_post_activation_max_read_readvariableop<
8savev2_quant_dense_56_optimizer_step_read_readvariableop8
4savev2_quant_dense_56_kernel_min_read_readvariableop8
4savev2_quant_dense_56_kernel_max_read_readvariableopA
=savev2_quant_dense_56_post_activation_min_read_readvariableopA
=savev2_quant_dense_56_post_activation_max_read_readvariableop<
8savev2_quant_dense_57_optimizer_step_read_readvariableop8
4savev2_quant_dense_57_kernel_min_read_readvariableop8
4savev2_quant_dense_57_kernel_max_read_readvariableopA
=savev2_quant_dense_57_post_activation_min_read_readvariableopA
=savev2_quant_dense_57_post_activation_max_read_readvariableop<
8savev2_quant_dense_58_optimizer_step_read_readvariableop8
4savev2_quant_dense_58_kernel_min_read_readvariableop8
4savev2_quant_dense_58_kernel_max_read_readvariableopA
=savev2_quant_dense_58_post_activation_min_read_readvariableopA
=savev2_quant_dense_58_post_activation_max_read_readvariableop<
8savev2_quant_dense_59_optimizer_step_read_readvariableop8
4savev2_quant_dense_59_kernel_min_read_readvariableop8
4savev2_quant_dense_59_kernel_max_read_readvariableopA
=savev2_quant_dense_59_post_activation_min_read_readvariableopA
=savev2_quant_dense_59_post_activation_max_read_readvariableop<
8savev2_quant_dense_60_optimizer_step_read_readvariableop8
4savev2_quant_dense_60_kernel_min_read_readvariableop8
4savev2_quant_dense_60_kernel_max_read_readvariableopA
=savev2_quant_dense_60_post_activation_min_read_readvariableopA
=savev2_quant_dense_60_post_activation_max_read_readvariableop<
8savev2_quant_dense_61_optimizer_step_read_readvariableop8
4savev2_quant_dense_61_kernel_min_read_readvariableop8
4savev2_quant_dense_61_kernel_max_read_readvariableopA
=savev2_quant_dense_61_post_activation_min_read_readvariableopA
=savev2_quant_dense_61_post_activation_max_read_readvariableop<
8savev2_quant_dense_62_optimizer_step_read_readvariableop8
4savev2_quant_dense_62_kernel_min_read_readvariableop8
4savev2_quant_dense_62_kernel_max_read_readvariableopA
=savev2_quant_dense_62_post_activation_min_read_readvariableopA
=savev2_quant_dense_62_post_activation_max_read_readvariableop<
8savev2_quant_dense_63_optimizer_step_read_readvariableop8
4savev2_quant_dense_63_kernel_min_read_readvariableop8
4savev2_quant_dense_63_kernel_max_read_readvariableopA
=savev2_quant_dense_63_post_activation_min_read_readvariableopA
=savev2_quant_dense_63_post_activation_max_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_dense_53_kernel_read_readvariableop,
(savev2_dense_53_bias_read_readvariableop.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop.
*savev2_dense_55_kernel_read_readvariableop,
(savev2_dense_55_bias_read_readvariableop.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop.
*savev2_dense_58_kernel_read_readvariableop,
(savev2_dense_58_bias_read_readvariableop.
*savev2_dense_59_kernel_read_readvariableop,
(savev2_dense_59_bias_read_readvariableop.
*savev2_dense_60_kernel_read_readvariableop,
(savev2_dense_60_bias_read_readvariableop.
*savev2_dense_61_kernel_read_readvariableop,
(savev2_dense_61_bias_read_readvariableop.
*savev2_dense_62_kernel_read_readvariableop,
(savev2_dense_62_bias_read_readvariableop.
*savev2_dense_63_kernel_read_readvariableop,
(savev2_dense_63_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_53_kernel_m_read_readvariableop3
/savev2_adam_dense_53_bias_m_read_readvariableop5
1savev2_adam_dense_54_kernel_m_read_readvariableop3
/savev2_adam_dense_54_bias_m_read_readvariableop5
1savev2_adam_dense_55_kernel_m_read_readvariableop3
/savev2_adam_dense_55_bias_m_read_readvariableop5
1savev2_adam_dense_56_kernel_m_read_readvariableop3
/savev2_adam_dense_56_bias_m_read_readvariableop5
1savev2_adam_dense_57_kernel_m_read_readvariableop3
/savev2_adam_dense_57_bias_m_read_readvariableop5
1savev2_adam_dense_58_kernel_m_read_readvariableop3
/savev2_adam_dense_58_bias_m_read_readvariableop5
1savev2_adam_dense_59_kernel_m_read_readvariableop3
/savev2_adam_dense_59_bias_m_read_readvariableop5
1savev2_adam_dense_60_kernel_m_read_readvariableop3
/savev2_adam_dense_60_bias_m_read_readvariableop5
1savev2_adam_dense_61_kernel_m_read_readvariableop3
/savev2_adam_dense_61_bias_m_read_readvariableop5
1savev2_adam_dense_62_kernel_m_read_readvariableop3
/savev2_adam_dense_62_bias_m_read_readvariableop5
1savev2_adam_dense_63_kernel_m_read_readvariableop3
/savev2_adam_dense_63_bias_m_read_readvariableop5
1savev2_adam_dense_53_kernel_v_read_readvariableop3
/savev2_adam_dense_53_bias_v_read_readvariableop5
1savev2_adam_dense_54_kernel_v_read_readvariableop3
/savev2_adam_dense_54_bias_v_read_readvariableop5
1savev2_adam_dense_55_kernel_v_read_readvariableop3
/savev2_adam_dense_55_bias_v_read_readvariableop5
1savev2_adam_dense_56_kernel_v_read_readvariableop3
/savev2_adam_dense_56_bias_v_read_readvariableop5
1savev2_adam_dense_57_kernel_v_read_readvariableop3
/savev2_adam_dense_57_bias_v_read_readvariableop5
1savev2_adam_dense_58_kernel_v_read_readvariableop3
/savev2_adam_dense_58_bias_v_read_readvariableop5
1savev2_adam_dense_59_kernel_v_read_readvariableop3
/savev2_adam_dense_59_bias_v_read_readvariableop5
1savev2_adam_dense_60_kernel_v_read_readvariableop3
/savev2_adam_dense_60_bias_v_read_readvariableop5
1savev2_adam_dense_61_kernel_v_read_readvariableop3
/savev2_adam_dense_61_bias_v_read_readvariableop5
1savev2_adam_dense_62_kernel_v_read_readvariableop3
/savev2_adam_dense_62_bias_v_read_readvariableop5
1savev2_adam_dense_63_kernel_v_read_readvariableop3
/savev2_adam_dense_63_bias_v_read_readvariableop
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
: �?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�?
value�?B�?�BElayer_with_weights-0/quantize_layer_15_min/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/quantize_layer_15_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-8/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-8/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-9/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-9/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-10/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-10/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-11/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-11/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/52/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/53/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/59/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/60/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/66/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/67/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/73/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/74/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/52/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/53/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/59/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/60/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/66/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/67/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/73/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/74/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �7
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Bsavev2_quantize_layer_15_quantize_layer_15_min_read_readvariableopBsavev2_quantize_layer_15_quantize_layer_15_max_read_readvariableop;savev2_quantize_layer_15_optimizer_step_read_readvariableop8savev2_quant_dense_53_optimizer_step_read_readvariableop4savev2_quant_dense_53_kernel_min_read_readvariableop4savev2_quant_dense_53_kernel_max_read_readvariableop=savev2_quant_dense_53_post_activation_min_read_readvariableop=savev2_quant_dense_53_post_activation_max_read_readvariableop8savev2_quant_dense_54_optimizer_step_read_readvariableop4savev2_quant_dense_54_kernel_min_read_readvariableop4savev2_quant_dense_54_kernel_max_read_readvariableop=savev2_quant_dense_54_post_activation_min_read_readvariableop=savev2_quant_dense_54_post_activation_max_read_readvariableop8savev2_quant_dense_55_optimizer_step_read_readvariableop4savev2_quant_dense_55_kernel_min_read_readvariableop4savev2_quant_dense_55_kernel_max_read_readvariableop=savev2_quant_dense_55_post_activation_min_read_readvariableop=savev2_quant_dense_55_post_activation_max_read_readvariableop8savev2_quant_dense_56_optimizer_step_read_readvariableop4savev2_quant_dense_56_kernel_min_read_readvariableop4savev2_quant_dense_56_kernel_max_read_readvariableop=savev2_quant_dense_56_post_activation_min_read_readvariableop=savev2_quant_dense_56_post_activation_max_read_readvariableop8savev2_quant_dense_57_optimizer_step_read_readvariableop4savev2_quant_dense_57_kernel_min_read_readvariableop4savev2_quant_dense_57_kernel_max_read_readvariableop=savev2_quant_dense_57_post_activation_min_read_readvariableop=savev2_quant_dense_57_post_activation_max_read_readvariableop8savev2_quant_dense_58_optimizer_step_read_readvariableop4savev2_quant_dense_58_kernel_min_read_readvariableop4savev2_quant_dense_58_kernel_max_read_readvariableop=savev2_quant_dense_58_post_activation_min_read_readvariableop=savev2_quant_dense_58_post_activation_max_read_readvariableop8savev2_quant_dense_59_optimizer_step_read_readvariableop4savev2_quant_dense_59_kernel_min_read_readvariableop4savev2_quant_dense_59_kernel_max_read_readvariableop=savev2_quant_dense_59_post_activation_min_read_readvariableop=savev2_quant_dense_59_post_activation_max_read_readvariableop8savev2_quant_dense_60_optimizer_step_read_readvariableop4savev2_quant_dense_60_kernel_min_read_readvariableop4savev2_quant_dense_60_kernel_max_read_readvariableop=savev2_quant_dense_60_post_activation_min_read_readvariableop=savev2_quant_dense_60_post_activation_max_read_readvariableop8savev2_quant_dense_61_optimizer_step_read_readvariableop4savev2_quant_dense_61_kernel_min_read_readvariableop4savev2_quant_dense_61_kernel_max_read_readvariableop=savev2_quant_dense_61_post_activation_min_read_readvariableop=savev2_quant_dense_61_post_activation_max_read_readvariableop8savev2_quant_dense_62_optimizer_step_read_readvariableop4savev2_quant_dense_62_kernel_min_read_readvariableop4savev2_quant_dense_62_kernel_max_read_readvariableop=savev2_quant_dense_62_post_activation_min_read_readvariableop=savev2_quant_dense_62_post_activation_max_read_readvariableop8savev2_quant_dense_63_optimizer_step_read_readvariableop4savev2_quant_dense_63_kernel_min_read_readvariableop4savev2_quant_dense_63_kernel_max_read_readvariableop=savev2_quant_dense_63_post_activation_min_read_readvariableop=savev2_quant_dense_63_post_activation_max_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_dense_53_kernel_read_readvariableop(savev2_dense_53_bias_read_readvariableop*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop*savev2_dense_55_kernel_read_readvariableop(savev2_dense_55_bias_read_readvariableop*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop*savev2_dense_58_kernel_read_readvariableop(savev2_dense_58_bias_read_readvariableop*savev2_dense_59_kernel_read_readvariableop(savev2_dense_59_bias_read_readvariableop*savev2_dense_60_kernel_read_readvariableop(savev2_dense_60_bias_read_readvariableop*savev2_dense_61_kernel_read_readvariableop(savev2_dense_61_bias_read_readvariableop*savev2_dense_62_kernel_read_readvariableop(savev2_dense_62_bias_read_readvariableop*savev2_dense_63_kernel_read_readvariableop(savev2_dense_63_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_53_kernel_m_read_readvariableop/savev2_adam_dense_53_bias_m_read_readvariableop1savev2_adam_dense_54_kernel_m_read_readvariableop/savev2_adam_dense_54_bias_m_read_readvariableop1savev2_adam_dense_55_kernel_m_read_readvariableop/savev2_adam_dense_55_bias_m_read_readvariableop1savev2_adam_dense_56_kernel_m_read_readvariableop/savev2_adam_dense_56_bias_m_read_readvariableop1savev2_adam_dense_57_kernel_m_read_readvariableop/savev2_adam_dense_57_bias_m_read_readvariableop1savev2_adam_dense_58_kernel_m_read_readvariableop/savev2_adam_dense_58_bias_m_read_readvariableop1savev2_adam_dense_59_kernel_m_read_readvariableop/savev2_adam_dense_59_bias_m_read_readvariableop1savev2_adam_dense_60_kernel_m_read_readvariableop/savev2_adam_dense_60_bias_m_read_readvariableop1savev2_adam_dense_61_kernel_m_read_readvariableop/savev2_adam_dense_61_bias_m_read_readvariableop1savev2_adam_dense_62_kernel_m_read_readvariableop/savev2_adam_dense_62_bias_m_read_readvariableop1savev2_adam_dense_63_kernel_m_read_readvariableop/savev2_adam_dense_63_bias_m_read_readvariableop1savev2_adam_dense_53_kernel_v_read_readvariableop/savev2_adam_dense_53_bias_v_read_readvariableop1savev2_adam_dense_54_kernel_v_read_readvariableop/savev2_adam_dense_54_bias_v_read_readvariableop1savev2_adam_dense_55_kernel_v_read_readvariableop/savev2_adam_dense_55_bias_v_read_readvariableop1savev2_adam_dense_56_kernel_v_read_readvariableop/savev2_adam_dense_56_bias_v_read_readvariableop1savev2_adam_dense_57_kernel_v_read_readvariableop/savev2_adam_dense_57_bias_v_read_readvariableop1savev2_adam_dense_58_kernel_v_read_readvariableop/savev2_adam_dense_58_bias_v_read_readvariableop1savev2_adam_dense_59_kernel_v_read_readvariableop/savev2_adam_dense_59_bias_v_read_readvariableop1savev2_adam_dense_60_kernel_v_read_readvariableop/savev2_adam_dense_60_bias_v_read_readvariableop1savev2_adam_dense_61_kernel_v_read_readvariableop/savev2_adam_dense_61_bias_v_read_readvariableop1savev2_adam_dense_62_kernel_v_read_readvariableop/savev2_adam_dense_62_bias_v_read_readvariableop1savev2_adam_dense_63_kernel_v_read_readvariableop/savev2_adam_dense_63_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
�2�	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :@@:@:@2:2:2 : : :::::::::: : :  : : @:@: : :@@:@:@2:2:2 : : :::::::::: : :  : : @:@:@@:@:@2:2:2 : : :::::::::: : :  : : @:@: 2(
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :$@ 

_output_shapes

:@@: A

_output_shapes
:@:$B 

_output_shapes

:@2: C

_output_shapes
:2:$D 

_output_shapes

:2 : E

_output_shapes
: :$F 

_output_shapes

: : G

_output_shapes
::$H 

_output_shapes

:: I

_output_shapes
::$J 

_output_shapes

:: K

_output_shapes
::$L 

_output_shapes

:: M

_output_shapes
::$N 

_output_shapes

:: O

_output_shapes
::$P 

_output_shapes

: : Q

_output_shapes
: :$R 

_output_shapes

:  : S

_output_shapes
: :$T 

_output_shapes

: @: U

_output_shapes
:@:V

_output_shapes
: :W

_output_shapes
: :$X 

_output_shapes

:@@: Y

_output_shapes
:@:$Z 

_output_shapes

:@2: [

_output_shapes
:2:$\ 

_output_shapes

:2 : ]

_output_shapes
: :$^ 

_output_shapes

: : _

_output_shapes
::$` 

_output_shapes

:: a

_output_shapes
::$b 

_output_shapes

:: c

_output_shapes
::$d 

_output_shapes

:: e

_output_shapes
::$f 

_output_shapes

:: g

_output_shapes
::$h 

_output_shapes

: : i

_output_shapes
: :$j 

_output_shapes

:  : k

_output_shapes
: :$l 

_output_shapes

: @: m

_output_shapes
:@:$n 

_output_shapes

:@@: o

_output_shapes
:@:$p 

_output_shapes

:@2: q

_output_shapes
:2:$r 

_output_shapes

:2 : s

_output_shapes
: :$t 

_output_shapes

: : u

_output_shapes
::$v 

_output_shapes

:: w

_output_shapes
::$x 

_output_shapes

:: y

_output_shapes
::$z 

_output_shapes

:: {

_output_shapes
::$| 

_output_shapes

:: }

_output_shapes
::$~ 

_output_shapes

: : 

_output_shapes
: :%� 

_output_shapes

:  :!�

_output_shapes
: :%� 

_output_shapes

: @:!�

_output_shapes
:@:�

_output_shapes
: 
� 
�
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3750401

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:2 J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:2 *
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

:2 *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�Y
�
E__inference_model_15_layer_call_and_return_conditional_losses_3752764
input_16#
quantize_layer_15_3752615: #
quantize_layer_15_3752617: (
quant_dense_53_3752620:@@ 
quant_dense_53_3752622:  
quant_dense_53_3752624: $
quant_dense_53_3752626:@ 
quant_dense_53_3752628:  
quant_dense_53_3752630: (
quant_dense_54_3752633:@2 
quant_dense_54_3752635:  
quant_dense_54_3752637: $
quant_dense_54_3752639:2 
quant_dense_54_3752641:  
quant_dense_54_3752643: (
quant_dense_55_3752646:2  
quant_dense_55_3752648:  
quant_dense_55_3752650: $
quant_dense_55_3752652:  
quant_dense_55_3752654:  
quant_dense_55_3752656: (
quant_dense_56_3752659:  
quant_dense_56_3752661:  
quant_dense_56_3752663: $
quant_dense_56_3752665: 
quant_dense_56_3752667:  
quant_dense_56_3752669: (
quant_dense_57_3752672: 
quant_dense_57_3752674:  
quant_dense_57_3752676: $
quant_dense_57_3752678: 
quant_dense_57_3752680:  
quant_dense_57_3752682: (
quant_dense_58_3752685: 
quant_dense_58_3752687:  
quant_dense_58_3752689: $
quant_dense_58_3752691: 
quant_dense_58_3752693:  
quant_dense_58_3752695: (
quant_dense_59_3752698: 
quant_dense_59_3752700:  
quant_dense_59_3752702: $
quant_dense_59_3752704: 
quant_dense_59_3752706:  
quant_dense_59_3752708: (
quant_dense_60_3752711: 
quant_dense_60_3752713:  
quant_dense_60_3752715: $
quant_dense_60_3752717: 
quant_dense_60_3752719:  
quant_dense_60_3752721: (
quant_dense_61_3752724:  
quant_dense_61_3752726:  
quant_dense_61_3752728: $
quant_dense_61_3752730:  
quant_dense_61_3752732:  
quant_dense_61_3752734: (
quant_dense_62_3752737:   
quant_dense_62_3752739:  
quant_dense_62_3752741: $
quant_dense_62_3752743:  
quant_dense_62_3752745:  
quant_dense_62_3752747: (
quant_dense_63_3752750: @ 
quant_dense_63_3752752:  
quant_dense_63_3752754: $
quant_dense_63_3752756:@ 
quant_dense_63_3752758:  
quant_dense_63_3752760: 
identity��&quant_dense_53/StatefulPartitionedCall�&quant_dense_54/StatefulPartitionedCall�&quant_dense_55/StatefulPartitionedCall�&quant_dense_56/StatefulPartitionedCall�&quant_dense_57/StatefulPartitionedCall�&quant_dense_58/StatefulPartitionedCall�&quant_dense_59/StatefulPartitionedCall�&quant_dense_60/StatefulPartitionedCall�&quant_dense_61/StatefulPartitionedCall�&quant_dense_62/StatefulPartitionedCall�&quant_dense_63/StatefulPartitionedCall�)quantize_layer_15/StatefulPartitionedCall�
)quantize_layer_15/StatefulPartitionedCallStatefulPartitionedCallinput_16quantize_layer_15_3752615quantize_layer_15_3752617*
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
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3751878�
&quant_dense_53/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_15/StatefulPartitionedCall:output:0quant_dense_53_3752620quant_dense_53_3752622quant_dense_53_3752624quant_dense_53_3752626quant_dense_53_3752628quant_dense_53_3752630*
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
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3751830�
&quant_dense_54/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_53/StatefulPartitionedCall:output:0quant_dense_54_3752633quant_dense_54_3752635quant_dense_54_3752637quant_dense_54_3752639quant_dense_54_3752641quant_dense_54_3752643*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3751738�
&quant_dense_55/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_54/StatefulPartitionedCall:output:0quant_dense_55_3752646quant_dense_55_3752648quant_dense_55_3752650quant_dense_55_3752652quant_dense_55_3752654quant_dense_55_3752656*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3751646�
&quant_dense_56/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_55/StatefulPartitionedCall:output:0quant_dense_56_3752659quant_dense_56_3752661quant_dense_56_3752663quant_dense_56_3752665quant_dense_56_3752667quant_dense_56_3752669*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3751554�
&quant_dense_57/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_56/StatefulPartitionedCall:output:0quant_dense_57_3752672quant_dense_57_3752674quant_dense_57_3752676quant_dense_57_3752678quant_dense_57_3752680quant_dense_57_3752682*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3751462�
&quant_dense_58/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_57/StatefulPartitionedCall:output:0quant_dense_58_3752685quant_dense_58_3752687quant_dense_58_3752689quant_dense_58_3752691quant_dense_58_3752693quant_dense_58_3752695*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3751370�
&quant_dense_59/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_58/StatefulPartitionedCall:output:0quant_dense_59_3752698quant_dense_59_3752700quant_dense_59_3752702quant_dense_59_3752704quant_dense_59_3752706quant_dense_59_3752708*
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
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3751278�
&quant_dense_60/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_59/StatefulPartitionedCall:output:0quant_dense_60_3752711quant_dense_60_3752713quant_dense_60_3752715quant_dense_60_3752717quant_dense_60_3752719quant_dense_60_3752721*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3751186�
&quant_dense_61/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_60/StatefulPartitionedCall:output:0quant_dense_61_3752724quant_dense_61_3752726quant_dense_61_3752728quant_dense_61_3752730quant_dense_61_3752732quant_dense_61_3752734*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3751094�
&quant_dense_62/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_61/StatefulPartitionedCall:output:0quant_dense_62_3752737quant_dense_62_3752739quant_dense_62_3752741quant_dense_62_3752743quant_dense_62_3752745quant_dense_62_3752747*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3751002�
&quant_dense_63/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_62/StatefulPartitionedCall:output:0quant_dense_63_3752750quant_dense_63_3752752quant_dense_63_3752754quant_dense_63_3752756quant_dense_63_3752758quant_dense_63_3752760*
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
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3750910~
IdentityIdentity/quant_dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_53/StatefulPartitionedCall'^quant_dense_54/StatefulPartitionedCall'^quant_dense_55/StatefulPartitionedCall'^quant_dense_56/StatefulPartitionedCall'^quant_dense_57/StatefulPartitionedCall'^quant_dense_58/StatefulPartitionedCall'^quant_dense_59/StatefulPartitionedCall'^quant_dense_60/StatefulPartitionedCall'^quant_dense_61/StatefulPartitionedCall'^quant_dense_62/StatefulPartitionedCall'^quant_dense_63/StatefulPartitionedCall*^quantize_layer_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&quant_dense_53/StatefulPartitionedCall&quant_dense_53/StatefulPartitionedCall2P
&quant_dense_54/StatefulPartitionedCall&quant_dense_54/StatefulPartitionedCall2P
&quant_dense_55/StatefulPartitionedCall&quant_dense_55/StatefulPartitionedCall2P
&quant_dense_56/StatefulPartitionedCall&quant_dense_56/StatefulPartitionedCall2P
&quant_dense_57/StatefulPartitionedCall&quant_dense_57/StatefulPartitionedCall2P
&quant_dense_58/StatefulPartitionedCall&quant_dense_58/StatefulPartitionedCall2P
&quant_dense_59/StatefulPartitionedCall&quant_dense_59/StatefulPartitionedCall2P
&quant_dense_60/StatefulPartitionedCall&quant_dense_60/StatefulPartitionedCall2P
&quant_dense_61/StatefulPartitionedCall&quant_dense_61/StatefulPartitionedCall2P
&quant_dense_62/StatefulPartitionedCall&quant_dense_62/StatefulPartitionedCall2P
&quant_dense_63/StatefulPartitionedCall&quant_dense_63/StatefulPartitionedCall2V
)quantize_layer_15/StatefulPartitionedCall)quantize_layer_15/StatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_16
�U
�
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3751462

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
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
:���������
 
_user_specified_nameinputs
�
�
3__inference_quantize_layer_15_layer_call_fn_3754002

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
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3750304o
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
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3755161

inputs=
+lastvaluequant_rank_readvariableop_resource:  /
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:  *
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

:  *
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:  *
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

:  *
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

:  *
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

:  *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� h
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 20
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
:��������� 
 
_user_specified_nameinputs
��
�R
#__inference__traced_restore_3756090
file_prefixB
8assignvariableop_quantize_layer_15_quantize_layer_15_min: D
:assignvariableop_1_quantize_layer_15_quantize_layer_15_max: =
3assignvariableop_2_quantize_layer_15_optimizer_step: :
0assignvariableop_3_quant_dense_53_optimizer_step: 6
,assignvariableop_4_quant_dense_53_kernel_min: 6
,assignvariableop_5_quant_dense_53_kernel_max: ?
5assignvariableop_6_quant_dense_53_post_activation_min: ?
5assignvariableop_7_quant_dense_53_post_activation_max: :
0assignvariableop_8_quant_dense_54_optimizer_step: 6
,assignvariableop_9_quant_dense_54_kernel_min: 7
-assignvariableop_10_quant_dense_54_kernel_max: @
6assignvariableop_11_quant_dense_54_post_activation_min: @
6assignvariableop_12_quant_dense_54_post_activation_max: ;
1assignvariableop_13_quant_dense_55_optimizer_step: 7
-assignvariableop_14_quant_dense_55_kernel_min: 7
-assignvariableop_15_quant_dense_55_kernel_max: @
6assignvariableop_16_quant_dense_55_post_activation_min: @
6assignvariableop_17_quant_dense_55_post_activation_max: ;
1assignvariableop_18_quant_dense_56_optimizer_step: 7
-assignvariableop_19_quant_dense_56_kernel_min: 7
-assignvariableop_20_quant_dense_56_kernel_max: @
6assignvariableop_21_quant_dense_56_post_activation_min: @
6assignvariableop_22_quant_dense_56_post_activation_max: ;
1assignvariableop_23_quant_dense_57_optimizer_step: 7
-assignvariableop_24_quant_dense_57_kernel_min: 7
-assignvariableop_25_quant_dense_57_kernel_max: @
6assignvariableop_26_quant_dense_57_post_activation_min: @
6assignvariableop_27_quant_dense_57_post_activation_max: ;
1assignvariableop_28_quant_dense_58_optimizer_step: 7
-assignvariableop_29_quant_dense_58_kernel_min: 7
-assignvariableop_30_quant_dense_58_kernel_max: @
6assignvariableop_31_quant_dense_58_post_activation_min: @
6assignvariableop_32_quant_dense_58_post_activation_max: ;
1assignvariableop_33_quant_dense_59_optimizer_step: 7
-assignvariableop_34_quant_dense_59_kernel_min: 7
-assignvariableop_35_quant_dense_59_kernel_max: @
6assignvariableop_36_quant_dense_59_post_activation_min: @
6assignvariableop_37_quant_dense_59_post_activation_max: ;
1assignvariableop_38_quant_dense_60_optimizer_step: 7
-assignvariableop_39_quant_dense_60_kernel_min: 7
-assignvariableop_40_quant_dense_60_kernel_max: @
6assignvariableop_41_quant_dense_60_post_activation_min: @
6assignvariableop_42_quant_dense_60_post_activation_max: ;
1assignvariableop_43_quant_dense_61_optimizer_step: 7
-assignvariableop_44_quant_dense_61_kernel_min: 7
-assignvariableop_45_quant_dense_61_kernel_max: @
6assignvariableop_46_quant_dense_61_post_activation_min: @
6assignvariableop_47_quant_dense_61_post_activation_max: ;
1assignvariableop_48_quant_dense_62_optimizer_step: 7
-assignvariableop_49_quant_dense_62_kernel_min: 7
-assignvariableop_50_quant_dense_62_kernel_max: @
6assignvariableop_51_quant_dense_62_post_activation_min: @
6assignvariableop_52_quant_dense_62_post_activation_max: ;
1assignvariableop_53_quant_dense_63_optimizer_step: 7
-assignvariableop_54_quant_dense_63_kernel_min: 7
-assignvariableop_55_quant_dense_63_kernel_max: @
6assignvariableop_56_quant_dense_63_post_activation_min: @
6assignvariableop_57_quant_dense_63_post_activation_max: '
assignvariableop_58_adam_iter:	 )
assignvariableop_59_adam_beta_1: )
assignvariableop_60_adam_beta_2: (
assignvariableop_61_adam_decay: 0
&assignvariableop_62_adam_learning_rate: 5
#assignvariableop_63_dense_53_kernel:@@/
!assignvariableop_64_dense_53_bias:@5
#assignvariableop_65_dense_54_kernel:@2/
!assignvariableop_66_dense_54_bias:25
#assignvariableop_67_dense_55_kernel:2 /
!assignvariableop_68_dense_55_bias: 5
#assignvariableop_69_dense_56_kernel: /
!assignvariableop_70_dense_56_bias:5
#assignvariableop_71_dense_57_kernel:/
!assignvariableop_72_dense_57_bias:5
#assignvariableop_73_dense_58_kernel:/
!assignvariableop_74_dense_58_bias:5
#assignvariableop_75_dense_59_kernel:/
!assignvariableop_76_dense_59_bias:5
#assignvariableop_77_dense_60_kernel:/
!assignvariableop_78_dense_60_bias:5
#assignvariableop_79_dense_61_kernel: /
!assignvariableop_80_dense_61_bias: 5
#assignvariableop_81_dense_62_kernel:  /
!assignvariableop_82_dense_62_bias: 5
#assignvariableop_83_dense_63_kernel: @/
!assignvariableop_84_dense_63_bias:@#
assignvariableop_85_total: #
assignvariableop_86_count: <
*assignvariableop_87_adam_dense_53_kernel_m:@@6
(assignvariableop_88_adam_dense_53_bias_m:@<
*assignvariableop_89_adam_dense_54_kernel_m:@26
(assignvariableop_90_adam_dense_54_bias_m:2<
*assignvariableop_91_adam_dense_55_kernel_m:2 6
(assignvariableop_92_adam_dense_55_bias_m: <
*assignvariableop_93_adam_dense_56_kernel_m: 6
(assignvariableop_94_adam_dense_56_bias_m:<
*assignvariableop_95_adam_dense_57_kernel_m:6
(assignvariableop_96_adam_dense_57_bias_m:<
*assignvariableop_97_adam_dense_58_kernel_m:6
(assignvariableop_98_adam_dense_58_bias_m:<
*assignvariableop_99_adam_dense_59_kernel_m:7
)assignvariableop_100_adam_dense_59_bias_m:=
+assignvariableop_101_adam_dense_60_kernel_m:7
)assignvariableop_102_adam_dense_60_bias_m:=
+assignvariableop_103_adam_dense_61_kernel_m: 7
)assignvariableop_104_adam_dense_61_bias_m: =
+assignvariableop_105_adam_dense_62_kernel_m:  7
)assignvariableop_106_adam_dense_62_bias_m: =
+assignvariableop_107_adam_dense_63_kernel_m: @7
)assignvariableop_108_adam_dense_63_bias_m:@=
+assignvariableop_109_adam_dense_53_kernel_v:@@7
)assignvariableop_110_adam_dense_53_bias_v:@=
+assignvariableop_111_adam_dense_54_kernel_v:@27
)assignvariableop_112_adam_dense_54_bias_v:2=
+assignvariableop_113_adam_dense_55_kernel_v:2 7
)assignvariableop_114_adam_dense_55_bias_v: =
+assignvariableop_115_adam_dense_56_kernel_v: 7
)assignvariableop_116_adam_dense_56_bias_v:=
+assignvariableop_117_adam_dense_57_kernel_v:7
)assignvariableop_118_adam_dense_57_bias_v:=
+assignvariableop_119_adam_dense_58_kernel_v:7
)assignvariableop_120_adam_dense_58_bias_v:=
+assignvariableop_121_adam_dense_59_kernel_v:7
)assignvariableop_122_adam_dense_59_bias_v:=
+assignvariableop_123_adam_dense_60_kernel_v:7
)assignvariableop_124_adam_dense_60_bias_v:=
+assignvariableop_125_adam_dense_61_kernel_v: 7
)assignvariableop_126_adam_dense_61_bias_v: =
+assignvariableop_127_adam_dense_62_kernel_v:  7
)assignvariableop_128_adam_dense_62_bias_v: =
+assignvariableop_129_adam_dense_63_kernel_v: @7
)assignvariableop_130_adam_dense_63_bias_v:@
identity_132��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�?
value�?B�?�BElayer_with_weights-0/quantize_layer_15_min/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/quantize_layer_15_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-6/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-7/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-8/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-8/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-9/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-9/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-10/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-10/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-11/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-11/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/52/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/53/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/59/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/60/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/66/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/67/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/73/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/74/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/52/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/53/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/59/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/60/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/66/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/67/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/73/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/74/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp8assignvariableop_quantize_layer_15_quantize_layer_15_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp:assignvariableop_1_quantize_layer_15_quantize_layer_15_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp3assignvariableop_2_quantize_layer_15_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_quant_dense_53_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_quant_dense_53_kernel_minIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp,assignvariableop_5_quant_dense_53_kernel_maxIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp5assignvariableop_6_quant_dense_53_post_activation_minIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp5assignvariableop_7_quant_dense_53_post_activation_maxIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_quant_dense_54_optimizer_stepIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_quant_dense_54_kernel_minIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp-assignvariableop_10_quant_dense_54_kernel_maxIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp6assignvariableop_11_quant_dense_54_post_activation_minIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp6assignvariableop_12_quant_dense_54_post_activation_maxIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp1assignvariableop_13_quant_dense_55_optimizer_stepIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_quant_dense_55_kernel_minIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_quant_dense_55_kernel_maxIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_quant_dense_55_post_activation_minIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_quant_dense_55_post_activation_maxIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_quant_dense_56_optimizer_stepIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp-assignvariableop_19_quant_dense_56_kernel_minIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp-assignvariableop_20_quant_dense_56_kernel_maxIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_quant_dense_56_post_activation_minIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_quant_dense_56_post_activation_maxIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp1assignvariableop_23_quant_dense_57_optimizer_stepIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp-assignvariableop_24_quant_dense_57_kernel_minIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp-assignvariableop_25_quant_dense_57_kernel_maxIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp6assignvariableop_26_quant_dense_57_post_activation_minIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp6assignvariableop_27_quant_dense_57_post_activation_maxIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp1assignvariableop_28_quant_dense_58_optimizer_stepIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp-assignvariableop_29_quant_dense_58_kernel_minIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp-assignvariableop_30_quant_dense_58_kernel_maxIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp6assignvariableop_31_quant_dense_58_post_activation_minIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp6assignvariableop_32_quant_dense_58_post_activation_maxIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp1assignvariableop_33_quant_dense_59_optimizer_stepIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp-assignvariableop_34_quant_dense_59_kernel_minIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp-assignvariableop_35_quant_dense_59_kernel_maxIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp6assignvariableop_36_quant_dense_59_post_activation_minIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp6assignvariableop_37_quant_dense_59_post_activation_maxIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp1assignvariableop_38_quant_dense_60_optimizer_stepIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp-assignvariableop_39_quant_dense_60_kernel_minIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp-assignvariableop_40_quant_dense_60_kernel_maxIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_quant_dense_60_post_activation_minIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp6assignvariableop_42_quant_dense_60_post_activation_maxIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp1assignvariableop_43_quant_dense_61_optimizer_stepIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp-assignvariableop_44_quant_dense_61_kernel_minIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp-assignvariableop_45_quant_dense_61_kernel_maxIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp6assignvariableop_46_quant_dense_61_post_activation_minIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp6assignvariableop_47_quant_dense_61_post_activation_maxIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp1assignvariableop_48_quant_dense_62_optimizer_stepIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp-assignvariableop_49_quant_dense_62_kernel_minIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp-assignvariableop_50_quant_dense_62_kernel_maxIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp6assignvariableop_51_quant_dense_62_post_activation_minIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp6assignvariableop_52_quant_dense_62_post_activation_maxIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp1assignvariableop_53_quant_dense_63_optimizer_stepIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp-assignvariableop_54_quant_dense_63_kernel_minIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp-assignvariableop_55_quant_dense_63_kernel_maxIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp6assignvariableop_56_quant_dense_63_post_activation_minIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp6assignvariableop_57_quant_dense_63_post_activation_maxIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpassignvariableop_58_adam_iterIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpassignvariableop_59_adam_beta_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpassignvariableop_60_adam_beta_2Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpassignvariableop_61_adam_decayIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp&assignvariableop_62_adam_learning_rateIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp#assignvariableop_63_dense_53_kernelIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp!assignvariableop_64_dense_53_biasIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp#assignvariableop_65_dense_54_kernelIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp!assignvariableop_66_dense_54_biasIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp#assignvariableop_67_dense_55_kernelIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp!assignvariableop_68_dense_55_biasIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp#assignvariableop_69_dense_56_kernelIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp!assignvariableop_70_dense_56_biasIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp#assignvariableop_71_dense_57_kernelIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp!assignvariableop_72_dense_57_biasIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp#assignvariableop_73_dense_58_kernelIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp!assignvariableop_74_dense_58_biasIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp#assignvariableop_75_dense_59_kernelIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp!assignvariableop_76_dense_59_biasIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp#assignvariableop_77_dense_60_kernelIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp!assignvariableop_78_dense_60_biasIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp#assignvariableop_79_dense_61_kernelIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp!assignvariableop_80_dense_61_biasIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp#assignvariableop_81_dense_62_kernelIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp!assignvariableop_82_dense_62_biasIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp#assignvariableop_83_dense_63_kernelIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp!assignvariableop_84_dense_63_biasIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOpassignvariableop_85_totalIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpassignvariableop_86_countIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_dense_53_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_dense_53_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_dense_54_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_dense_54_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_dense_55_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_dense_55_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_dense_56_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_dense_56_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_dense_57_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_dense_57_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_dense_58_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_dense_58_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp*assignvariableop_99_adam_dense_59_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp)assignvariableop_100_adam_dense_59_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_dense_60_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_dense_60_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_dense_61_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_dense_61_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp+assignvariableop_105_adam_dense_62_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp)assignvariableop_106_adam_dense_62_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp+assignvariableop_107_adam_dense_63_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp)assignvariableop_108_adam_dense_63_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp+assignvariableop_109_adam_dense_53_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp)assignvariableop_110_adam_dense_53_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_dense_54_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_dense_54_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp+assignvariableop_113_adam_dense_55_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp)assignvariableop_114_adam_dense_55_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp+assignvariableop_115_adam_dense_56_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp)assignvariableop_116_adam_dense_56_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp+assignvariableop_117_adam_dense_57_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp)assignvariableop_118_adam_dense_57_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOp+assignvariableop_119_adam_dense_58_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOp)assignvariableop_120_adam_dense_58_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOp+assignvariableop_121_adam_dense_59_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOp)assignvariableop_122_adam_dense_59_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_123AssignVariableOp+assignvariableop_123_adam_dense_60_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_124AssignVariableOp)assignvariableop_124_adam_dense_60_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_125AssignVariableOp+assignvariableop_125_adam_dense_61_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_126AssignVariableOp)assignvariableop_126_adam_dense_61_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_127AssignVariableOp+assignvariableop_127_adam_dense_62_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_128AssignVariableOp)assignvariableop_128_adam_dense_62_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_129AssignVariableOp+assignvariableop_129_adam_dense_63_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_130AssignVariableOp)assignvariableop_130_adam_dense_63_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_131Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_132IdentityIdentity_131:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_132Identity_132:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302*
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�Y
�
E__inference_model_15_layer_call_and_return_conditional_losses_3750695

inputs#
quantize_layer_15_3750305: #
quantize_layer_15_3750307: (
quant_dense_53_3750332:@@ 
quant_dense_53_3750334:  
quant_dense_53_3750336: $
quant_dense_53_3750338:@ 
quant_dense_53_3750340:  
quant_dense_53_3750342: (
quant_dense_54_3750367:@2 
quant_dense_54_3750369:  
quant_dense_54_3750371: $
quant_dense_54_3750373:2 
quant_dense_54_3750375:  
quant_dense_54_3750377: (
quant_dense_55_3750402:2  
quant_dense_55_3750404:  
quant_dense_55_3750406: $
quant_dense_55_3750408:  
quant_dense_55_3750410:  
quant_dense_55_3750412: (
quant_dense_56_3750437:  
quant_dense_56_3750439:  
quant_dense_56_3750441: $
quant_dense_56_3750443: 
quant_dense_56_3750445:  
quant_dense_56_3750447: (
quant_dense_57_3750472: 
quant_dense_57_3750474:  
quant_dense_57_3750476: $
quant_dense_57_3750478: 
quant_dense_57_3750480:  
quant_dense_57_3750482: (
quant_dense_58_3750507: 
quant_dense_58_3750509:  
quant_dense_58_3750511: $
quant_dense_58_3750513: 
quant_dense_58_3750515:  
quant_dense_58_3750517: (
quant_dense_59_3750542: 
quant_dense_59_3750544:  
quant_dense_59_3750546: $
quant_dense_59_3750548: 
quant_dense_59_3750550:  
quant_dense_59_3750552: (
quant_dense_60_3750577: 
quant_dense_60_3750579:  
quant_dense_60_3750581: $
quant_dense_60_3750583: 
quant_dense_60_3750585:  
quant_dense_60_3750587: (
quant_dense_61_3750612:  
quant_dense_61_3750614:  
quant_dense_61_3750616: $
quant_dense_61_3750618:  
quant_dense_61_3750620:  
quant_dense_61_3750622: (
quant_dense_62_3750647:   
quant_dense_62_3750649:  
quant_dense_62_3750651: $
quant_dense_62_3750653:  
quant_dense_62_3750655:  
quant_dense_62_3750657: (
quant_dense_63_3750681: @ 
quant_dense_63_3750683:  
quant_dense_63_3750685: $
quant_dense_63_3750687:@ 
quant_dense_63_3750689:  
quant_dense_63_3750691: 
identity��&quant_dense_53/StatefulPartitionedCall�&quant_dense_54/StatefulPartitionedCall�&quant_dense_55/StatefulPartitionedCall�&quant_dense_56/StatefulPartitionedCall�&quant_dense_57/StatefulPartitionedCall�&quant_dense_58/StatefulPartitionedCall�&quant_dense_59/StatefulPartitionedCall�&quant_dense_60/StatefulPartitionedCall�&quant_dense_61/StatefulPartitionedCall�&quant_dense_62/StatefulPartitionedCall�&quant_dense_63/StatefulPartitionedCall�)quantize_layer_15/StatefulPartitionedCall�
)quantize_layer_15/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_15_3750305quantize_layer_15_3750307*
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
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3750304�
&quant_dense_53/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_15/StatefulPartitionedCall:output:0quant_dense_53_3750332quant_dense_53_3750334quant_dense_53_3750336quant_dense_53_3750338quant_dense_53_3750340quant_dense_53_3750342*
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
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3750331�
&quant_dense_54/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_53/StatefulPartitionedCall:output:0quant_dense_54_3750367quant_dense_54_3750369quant_dense_54_3750371quant_dense_54_3750373quant_dense_54_3750375quant_dense_54_3750377*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3750366�
&quant_dense_55/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_54/StatefulPartitionedCall:output:0quant_dense_55_3750402quant_dense_55_3750404quant_dense_55_3750406quant_dense_55_3750408quant_dense_55_3750410quant_dense_55_3750412*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3750401�
&quant_dense_56/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_55/StatefulPartitionedCall:output:0quant_dense_56_3750437quant_dense_56_3750439quant_dense_56_3750441quant_dense_56_3750443quant_dense_56_3750445quant_dense_56_3750447*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3750436�
&quant_dense_57/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_56/StatefulPartitionedCall:output:0quant_dense_57_3750472quant_dense_57_3750474quant_dense_57_3750476quant_dense_57_3750478quant_dense_57_3750480quant_dense_57_3750482*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3750471�
&quant_dense_58/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_57/StatefulPartitionedCall:output:0quant_dense_58_3750507quant_dense_58_3750509quant_dense_58_3750511quant_dense_58_3750513quant_dense_58_3750515quant_dense_58_3750517*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3750506�
&quant_dense_59/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_58/StatefulPartitionedCall:output:0quant_dense_59_3750542quant_dense_59_3750544quant_dense_59_3750546quant_dense_59_3750548quant_dense_59_3750550quant_dense_59_3750552*
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
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3750541�
&quant_dense_60/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_59/StatefulPartitionedCall:output:0quant_dense_60_3750577quant_dense_60_3750579quant_dense_60_3750581quant_dense_60_3750583quant_dense_60_3750585quant_dense_60_3750587*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3750576�
&quant_dense_61/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_60/StatefulPartitionedCall:output:0quant_dense_61_3750612quant_dense_61_3750614quant_dense_61_3750616quant_dense_61_3750618quant_dense_61_3750620quant_dense_61_3750622*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3750611�
&quant_dense_62/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_61/StatefulPartitionedCall:output:0quant_dense_62_3750647quant_dense_62_3750649quant_dense_62_3750651quant_dense_62_3750653quant_dense_62_3750655quant_dense_62_3750657*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3750646�
&quant_dense_63/StatefulPartitionedCallStatefulPartitionedCall/quant_dense_62/StatefulPartitionedCall:output:0quant_dense_63_3750681quant_dense_63_3750683quant_dense_63_3750685quant_dense_63_3750687quant_dense_63_3750689quant_dense_63_3750691*
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
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3750680~
IdentityIdentity/quant_dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_53/StatefulPartitionedCall'^quant_dense_54/StatefulPartitionedCall'^quant_dense_55/StatefulPartitionedCall'^quant_dense_56/StatefulPartitionedCall'^quant_dense_57/StatefulPartitionedCall'^quant_dense_58/StatefulPartitionedCall'^quant_dense_59/StatefulPartitionedCall'^quant_dense_60/StatefulPartitionedCall'^quant_dense_61/StatefulPartitionedCall'^quant_dense_62/StatefulPartitionedCall'^quant_dense_63/StatefulPartitionedCall*^quantize_layer_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&quant_dense_53/StatefulPartitionedCall&quant_dense_53/StatefulPartitionedCall2P
&quant_dense_54/StatefulPartitionedCall&quant_dense_54/StatefulPartitionedCall2P
&quant_dense_55/StatefulPartitionedCall&quant_dense_55/StatefulPartitionedCall2P
&quant_dense_56/StatefulPartitionedCall&quant_dense_56/StatefulPartitionedCall2P
&quant_dense_57/StatefulPartitionedCall&quant_dense_57/StatefulPartitionedCall2P
&quant_dense_58/StatefulPartitionedCall&quant_dense_58/StatefulPartitionedCall2P
&quant_dense_59/StatefulPartitionedCall&quant_dense_59/StatefulPartitionedCall2P
&quant_dense_60/StatefulPartitionedCall&quant_dense_60/StatefulPartitionedCall2P
&quant_dense_61/StatefulPartitionedCall&quant_dense_61/StatefulPartitionedCall2P
&quant_dense_62/StatefulPartitionedCall&quant_dense_62/StatefulPartitionedCall2P
&quant_dense_63/StatefulPartitionedCall&quant_dense_63/StatefulPartitionedCall2V
)quantize_layer_15/StatefulPartitionedCall)quantize_layer_15/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3754544

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3751094

inputs=
+lastvaluequant_rank_readvariableop_resource: /
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
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

: *
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
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

: *
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

: *
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

: *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� h
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
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
:���������
 
_user_specified_nameinputs
�
�
*__inference_model_15_layer_call_fn_3750834
input_16
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
	unknown_7:@2
	unknown_8: 
	unknown_9: 

unknown_10:2

unknown_11: 

unknown_12: 

unknown_13:2 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:

unknown_23: 

unknown_24: 

unknown_25:

unknown_26: 

unknown_27: 

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:

unknown_32: 

unknown_33: 

unknown_34:

unknown_35: 

unknown_36: 

unknown_37:

unknown_38: 

unknown_39: 

unknown_40:

unknown_41: 

unknown_42: 

unknown_43:

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: 

unknown_53: 

unknown_54: 

unknown_55:  

unknown_56: 

unknown_57: 

unknown_58: 

unknown_59: 

unknown_60: 

unknown_61: @

unknown_62: 

unknown_63: 

unknown_64:@

unknown_65: 

unknown_66: 
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_15_layer_call_and_return_conditional_losses_3750695o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_16
�U
�
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3754937

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
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
0__inference_quant_dense_62_layer_call_fn_3755066

inputs
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
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
:��������� *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3750646o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_model_15_layer_call_fn_3753195

inputs
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
	unknown_7:@2
	unknown_8: 
	unknown_9: 

unknown_10:2

unknown_11: 

unknown_12: 

unknown_13:2 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:

unknown_23: 

unknown_24: 

unknown_25:

unknown_26: 

unknown_27: 

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:

unknown_32: 

unknown_33: 

unknown_34:

unknown_35: 

unknown_36: 

unknown_37:

unknown_38: 

unknown_39: 

unknown_40:

unknown_41: 

unknown_42: 

unknown_43:

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: 

unknown_53: 

unknown_54: 

unknown_55:  

unknown_56: 

unknown_57: 

unknown_58: 

unknown_59: 

unknown_60: 

unknown_61: @

unknown_62: 

unknown_63: 

unknown_64:@

unknown_65: 

unknown_66: 
identity��StatefulPartitionedCall�	
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*8
_read_only_resource_inputs
	!$'*-0369<?B*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_15_layer_call_and_return_conditional_losses_3752180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3750366

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@2J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:2K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@2*
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

:@2*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2�
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
:���������2�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������2�
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
�U
�
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3754153

inputs=
+lastvaluequant_rank_readvariableop_resource:@@/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
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

:@@*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
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

:@@*
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

:@@*
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

:@@*
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
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@h
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
�
�
%__inference_signature_wrapper_3752913
input_16
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
	unknown_7:@2
	unknown_8: 
	unknown_9: 

unknown_10:2

unknown_11: 

unknown_12: 

unknown_13:2 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:

unknown_23: 

unknown_24: 

unknown_25:

unknown_26: 

unknown_27: 

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:

unknown_32: 

unknown_33: 

unknown_34:

unknown_35: 

unknown_36: 

unknown_37:

unknown_38: 

unknown_39: 

unknown_40:

unknown_41: 

unknown_42: 

unknown_43:

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: 

unknown_53: 

unknown_54: 

unknown_55:  

unknown_56: 

unknown_57: 

unknown_58: 

unknown_59: 

unknown_60: 

unknown_61: @

unknown_62: 

unknown_63: 

unknown_64:@

unknown_65: 

unknown_66: 
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_3750288o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_16
�
�
0__inference_quant_dense_60_layer_call_fn_3754859

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
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
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3751186o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
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
� 
�
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3754656

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3750541

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
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

:*
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
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_model_15_layer_call_fn_3753054

inputs
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
	unknown_7:@2
	unknown_8: 
	unknown_9: 

unknown_10:2

unknown_11: 

unknown_12: 

unknown_13:2 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22:

unknown_23: 

unknown_24: 

unknown_25:

unknown_26: 

unknown_27: 

unknown_28:

unknown_29: 

unknown_30: 

unknown_31:

unknown_32: 

unknown_33: 

unknown_34:

unknown_35: 

unknown_36: 

unknown_37:

unknown_38: 

unknown_39: 

unknown_40:

unknown_41: 

unknown_42: 

unknown_43:

unknown_44: 

unknown_45: 

unknown_46:

unknown_47: 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: 

unknown_53: 

unknown_54: 

unknown_55:  

unknown_56: 

unknown_57: 

unknown_58: 

unknown_59: 

unknown_60: 

unknown_61: @

unknown_62: 

unknown_63: 

unknown_64:@

unknown_65: 

unknown_66: 
identity��StatefulPartitionedCall�	
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_15_layer_call_and_return_conditional_losses_3750695o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_53_layer_call_fn_3754075

inputs
unknown:@@
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
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3751830o
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
:���������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�"
�
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3751878

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
�U
�
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3754489

inputs=
+lastvaluequant_rank_readvariableop_resource: /
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
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

: *
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
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

: *
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

: *
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

: *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������h
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 20
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
:��������� 
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3754265

inputs=
+lastvaluequant_rank_readvariableop_resource:@2/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:2@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@2*
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

:@2*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@2*
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

:@2*
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

:@2*
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

:@2*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2h
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
:���������2�
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������2�
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
�
�
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3754020

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
�
�
0__inference_quant_dense_53_layer_call_fn_3754058

inputs
unknown:@@
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
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3750331o
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
:���������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3750506

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3750304

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
�U
�
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3754825

inputs=
+lastvaluequant_rank_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
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

:*
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

:*
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

:*
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
:���������: : : : : : 20
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
:���������
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3751830

inputs=
+lastvaluequant_rank_readvariableop_resource:@@/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
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

:@@*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
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

:@@*
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

:@@*
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

:@@*
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
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@h
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
�
�
0__inference_quant_dense_63_layer_call_fn_3755195

inputs
unknown: @
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
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3750910o
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
:��������� : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_56_layer_call_fn_3754394

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2:
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
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3750436o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_59_layer_call_fn_3754747

inputs
unknown:
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
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3751278o
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
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�S
E__inference_model_15_layer_call_and_return_conditional_losses_3753390

inputs]
Squantize_layer_15_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: _
Uquantize_layer_15_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@@Y
Oquant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_53_biasadd_readvariableop_resource:@Z
Pquant_dense_53_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_53_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@2Y
Oquant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_54_biasadd_readvariableop_resource:2Z
Pquant_dense_54_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_54_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:2 Y
Oquant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_55_biasadd_readvariableop_resource: Z
Pquant_dense_55_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_55_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: Y
Oquant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_56_biasadd_readvariableop_resource:Z
Pquant_dense_56_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_56_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:Y
Oquant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_57_biasadd_readvariableop_resource:Z
Pquant_dense_57_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_57_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:Y
Oquant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_58_biasadd_readvariableop_resource:Z
Pquant_dense_58_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_58_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:Y
Oquant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_59_biasadd_readvariableop_resource:Z
Pquant_dense_59_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_59_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:Y
Oquant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_60_biasadd_readvariableop_resource:Z
Pquant_dense_60_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_60_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: Y
Oquant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_61_biasadd_readvariableop_resource: Z
Pquant_dense_61_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_61_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:  Y
Oquant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_62_biasadd_readvariableop_resource: Z
Pquant_dense_62_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_62_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: @Y
Oquant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_63_biasadd_readvariableop_resource:@Z
Pquant_dense_63_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_63_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��%quant_dense_53/BiasAdd/ReadVariableOp�Dquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_54/BiasAdd/ReadVariableOp�Dquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_55/BiasAdd/ReadVariableOp�Dquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_56/BiasAdd/ReadVariableOp�Dquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_57/BiasAdd/ReadVariableOp�Dquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_58/BiasAdd/ReadVariableOp�Dquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_59/BiasAdd/ReadVariableOp�Dquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_60/BiasAdd/ReadVariableOp�Dquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_61/BiasAdd/ReadVariableOp�Dquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_62/BiasAdd/ReadVariableOp�Dquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�%quant_dense_63/BiasAdd/ReadVariableOp�Dquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Jquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Lquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Jquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpSquantize_layer_15_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Lquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpUquantize_layer_15_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
;quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsRquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Tquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Dquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@@*
narrow_range(�
quant_dense_53/MatMulMatMulEquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
%quant_dense_53/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_53_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_53/BiasAddBiasAddquant_dense_53/MatMul:product:0-quant_dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@n
quant_dense_53/ReluReluquant_dense_53/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
Gquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_53_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_53_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_53/Relu:activations:0Oquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Dquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@2*
dtype0�
Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@2*
narrow_range(�
quant_dense_54/MatMulMatMulBquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������2�
%quant_dense_54/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_54_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
quant_dense_54/BiasAddBiasAddquant_dense_54/MatMul:product:0-quant_dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2n
quant_dense_54/ReluReluquant_dense_54/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
Gquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_54_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_54_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_54/Relu:activations:0Oquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������2�
Dquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:2 *
dtype0�
Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:2 *
narrow_range(�
quant_dense_55/MatMulMatMulBquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� �
%quant_dense_55/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_55_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
quant_dense_55/BiasAddBiasAddquant_dense_55/MatMul:product:0-quant_dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
quant_dense_55/ReluReluquant_dense_55/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
Gquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_55_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_55_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_55/Relu:activations:0Oquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:��������� �
Dquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: *
dtype0�
Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

: *
narrow_range(�
quant_dense_56/MatMulMatMulBquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_56/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_56/BiasAddBiasAddquant_dense_56/MatMul:product:0-quant_dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_56/ReluReluquant_dense_56/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Gquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_56_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_56_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_56/Relu:activations:0Oquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Dquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_57/MatMulMatMulBquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_57/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_57/BiasAddBiasAddquant_dense_57/MatMul:product:0-quant_dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_57/ReluReluquant_dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Gquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_57_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_57_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_57/Relu:activations:0Oquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Dquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_58/MatMulMatMulBquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_58/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_58/BiasAddBiasAddquant_dense_58/MatMul:product:0-quant_dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_58/ReluReluquant_dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Gquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_58_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_58_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_58/Relu:activations:0Oquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Dquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_59/MatMulMatMulBquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_59/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_59/BiasAddBiasAddquant_dense_59/MatMul:product:0-quant_dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_59/ReluReluquant_dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Gquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_59_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_59_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_59/Relu:activations:0Oquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Dquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_60/MatMulMatMulBquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
%quant_dense_60/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_60/BiasAddBiasAddquant_dense_60/MatMul:product:0-quant_dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
quant_dense_60/ReluReluquant_dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Gquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_60_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_60_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_60/Relu:activations:0Oquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Dquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: *
dtype0�
Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

: *
narrow_range(�
quant_dense_61/MatMulMatMulBquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� �
%quant_dense_61/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
quant_dense_61/BiasAddBiasAddquant_dense_61/MatMul:product:0-quant_dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
quant_dense_61/ReluReluquant_dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
Gquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_61_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_61_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_61/Relu:activations:0Oquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:��������� �
Dquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:  *
dtype0�
Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:  *
narrow_range(�
quant_dense_62/MatMulMatMulBquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� �
%quant_dense_62/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_62_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
quant_dense_62/BiasAddBiasAddquant_dense_62/MatMul:product:0-quant_dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� n
quant_dense_62/ReluReluquant_dense_62/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
Gquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_62_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_62_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_dense_62/Relu:activations:0Oquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:��������� �
Dquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: @*
dtype0�
Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

: @*
narrow_range(�
quant_dense_63/MatMulMatMulBquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
%quant_dense_63/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_63_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_63/BiasAddBiasAddquant_dense_63/MatMul:product:0-quant_dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Gquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_63_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_63_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_63/BiasAdd:output:0Oquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityBquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�$
NoOpNoOp&^quant_dense_53/BiasAdd/ReadVariableOpE^quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_54/BiasAdd/ReadVariableOpE^quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_55/BiasAdd/ReadVariableOpE^quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_56/BiasAdd/ReadVariableOpE^quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_57/BiasAdd/ReadVariableOpE^quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_58/BiasAdd/ReadVariableOpE^quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_59/BiasAdd/ReadVariableOpE^quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_60/BiasAdd/ReadVariableOpE^quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_61/BiasAdd/ReadVariableOpE^quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_62/BiasAdd/ReadVariableOpE^quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1&^quant_dense_63/BiasAdd/ReadVariableOpE^quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1K^quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpM^quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%quant_dense_53/BiasAdd/ReadVariableOp%quant_dense_53/BiasAdd/ReadVariableOp2�
Dquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_54/BiasAdd/ReadVariableOp%quant_dense_54/BiasAdd/ReadVariableOp2�
Dquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_55/BiasAdd/ReadVariableOp%quant_dense_55/BiasAdd/ReadVariableOp2�
Dquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_56/BiasAdd/ReadVariableOp%quant_dense_56/BiasAdd/ReadVariableOp2�
Dquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_57/BiasAdd/ReadVariableOp%quant_dense_57/BiasAdd/ReadVariableOp2�
Dquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_58/BiasAdd/ReadVariableOp%quant_dense_58/BiasAdd/ReadVariableOp2�
Dquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_59/BiasAdd/ReadVariableOp%quant_dense_59/BiasAdd/ReadVariableOp2�
Dquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_60/BiasAdd/ReadVariableOp%quant_dense_60/BiasAdd/ReadVariableOp2�
Dquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_61/BiasAdd/ReadVariableOp%quant_dense_61/BiasAdd/ReadVariableOp2�
Dquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_62/BiasAdd/ReadVariableOp%quant_dense_62/BiasAdd/ReadVariableOp2�
Dquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12N
%quant_dense_63/BiasAdd/ReadVariableOp%quant_dense_63/BiasAdd/ReadVariableOp2�
Dquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Jquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Lquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Lquantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�T
�
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3755271

inputs=
+lastvaluequant_rank_readvariableop_resource: @/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: @*
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

: @*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: @*
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

: @*
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

: @*
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

: @*
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
:��������� : : : : : : 20
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
:��������� 
 
_user_specified_nameinputs
� 
�
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3750471

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
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

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_60_layer_call_fn_3754842

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
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
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3750576o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
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
� 
�
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3755104

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:  J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource: K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:  *
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

:  *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�U
�
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3751002

inputs=
+lastvaluequant_rank_readvariableop_resource:  /
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:  *
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

:  *
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:  *
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

:  *
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

:  *
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

:  *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� h
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : : : : : 20
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
:��������� 
 
_user_specified_nameinputs
��
�]
"__inference__wrapped_model_3750288
input_16f
\model_15_quantize_layer_15_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: h
^model_15_quantize_layer_15_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@@b
Xmodel_15_quant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_53_biasadd_readvariableop_resource:@c
Ymodel_15_quant_dense_53_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_53_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@2b
Xmodel_15_quant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_54_biasadd_readvariableop_resource:2c
Ymodel_15_quant_dense_54_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_54_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:2 b
Xmodel_15_quant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_55_biasadd_readvariableop_resource: c
Ymodel_15_quant_dense_55_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_55_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: b
Xmodel_15_quant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_56_biasadd_readvariableop_resource:c
Ymodel_15_quant_dense_56_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_56_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:b
Xmodel_15_quant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_57_biasadd_readvariableop_resource:c
Ymodel_15_quant_dense_57_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_57_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:b
Xmodel_15_quant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_58_biasadd_readvariableop_resource:c
Ymodel_15_quant_dense_58_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_58_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:b
Xmodel_15_quant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_59_biasadd_readvariableop_resource:c
Ymodel_15_quant_dense_59_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_59_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:b
Xmodel_15_quant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_60_biasadd_readvariableop_resource:c
Ymodel_15_quant_dense_60_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_60_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: b
Xmodel_15_quant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_61_biasadd_readvariableop_resource: c
Ymodel_15_quant_dense_61_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_61_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:  b
Xmodel_15_quant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_62_biasadd_readvariableop_resource: c
Ymodel_15_quant_dense_62_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_62_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_15_quant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: @b
Xmodel_15_quant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_15_quant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_15_quant_dense_63_biasadd_readvariableop_resource:@c
Ymodel_15_quant_dense_63_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_15_quant_dense_63_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��.model_15/quant_dense_53/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_54/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_55/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_56/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_57/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_58/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_59/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_60/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_61/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_62/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�.model_15/quant_dense_63/BiasAdd/ReadVariableOp�Mmodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Smodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Umodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Smodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp\model_15_quantize_layer_15_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Umodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp^model_15_quantize_layer_15_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Dmodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinput_16[model_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0]model_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Mmodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Omodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_53_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@@*
narrow_range(�
model_15/quant_dense_53/MatMulMatMulNmodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
.model_15/quant_dense_53/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_53_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_15/quant_dense_53/BiasAddBiasAdd(model_15/quant_dense_53/MatMul:product:06model_15/quant_dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
model_15/quant_dense_53/ReluRelu(model_15/quant_dense_53/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
Pmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_53_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_53_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_53/Relu:activations:0Xmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Mmodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@2*
dtype0�
Omodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_54_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@2*
narrow_range(�
model_15/quant_dense_54/MatMulMatMulKmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������2�
.model_15/quant_dense_54/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_54_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
model_15/quant_dense_54/BiasAddBiasAdd(model_15/quant_dense_54/MatMul:product:06model_15/quant_dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
model_15/quant_dense_54/ReluRelu(model_15/quant_dense_54/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
Pmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_54_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_54_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_54/Relu:activations:0Xmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������2�
Mmodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:2 *
dtype0�
Omodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_55_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:2 *
narrow_range(�
model_15/quant_dense_55/MatMulMatMulKmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� �
.model_15/quant_dense_55/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_55_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_15/quant_dense_55/BiasAddBiasAdd(model_15/quant_dense_55/MatMul:product:06model_15/quant_dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model_15/quant_dense_55/ReluRelu(model_15/quant_dense_55/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
Pmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_55_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_55_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_55/Relu:activations:0Xmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:��������� �
Mmodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: *
dtype0�
Omodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_56_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

: *
narrow_range(�
model_15/quant_dense_56/MatMulMatMulKmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
.model_15/quant_dense_56/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_15/quant_dense_56/BiasAddBiasAdd(model_15/quant_dense_56/MatMul:product:06model_15/quant_dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_15/quant_dense_56/ReluRelu(model_15/quant_dense_56/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_56_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_56_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_56/Relu:activations:0Xmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Mmodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Omodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_57_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
model_15/quant_dense_57/MatMulMatMulKmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
.model_15/quant_dense_57/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_15/quant_dense_57/BiasAddBiasAdd(model_15/quant_dense_57/MatMul:product:06model_15/quant_dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_15/quant_dense_57/ReluRelu(model_15/quant_dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_57_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_57_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_57/Relu:activations:0Xmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Mmodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Omodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_58_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
model_15/quant_dense_58/MatMulMatMulKmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
.model_15/quant_dense_58/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_15/quant_dense_58/BiasAddBiasAdd(model_15/quant_dense_58/MatMul:product:06model_15/quant_dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_15/quant_dense_58/ReluRelu(model_15/quant_dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_58_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_58_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_58/Relu:activations:0Xmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Mmodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Omodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_59_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
model_15/quant_dense_59/MatMulMatMulKmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
.model_15/quant_dense_59/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_15/quant_dense_59/BiasAddBiasAdd(model_15/quant_dense_59/MatMul:product:06model_15/quant_dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_15/quant_dense_59/ReluRelu(model_15/quant_dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_59_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_59_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_59/Relu:activations:0Xmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Mmodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Omodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_60_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
model_15/quant_dense_60/MatMulMatMulKmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
.model_15/quant_dense_60/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_15/quant_dense_60/BiasAddBiasAdd(model_15/quant_dense_60/MatMul:product:06model_15/quant_dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_15/quant_dense_60/ReluRelu(model_15/quant_dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_60_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_60_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_60/Relu:activations:0Xmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Mmodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: *
dtype0�
Omodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_61_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

: *
narrow_range(�
model_15/quant_dense_61/MatMulMatMulKmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� �
.model_15/quant_dense_61/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_61_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_15/quant_dense_61/BiasAddBiasAdd(model_15/quant_dense_61/MatMul:product:06model_15/quant_dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model_15/quant_dense_61/ReluRelu(model_15/quant_dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
Pmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_61_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_61_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_61/Relu:activations:0Xmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:��������� �
Mmodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:  *
dtype0�
Omodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_62_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:  *
narrow_range(�
model_15/quant_dense_62/MatMulMatMulKmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� �
.model_15/quant_dense_62/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_62_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_15/quant_dense_62/BiasAddBiasAdd(model_15/quant_dense_62/MatMul:product:06model_15/quant_dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model_15/quant_dense_62/ReluRelu(model_15/quant_dense_62/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
Pmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_62_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_62_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars*model_15/quant_dense_62/Relu:activations:0Xmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:��������� �
Mmodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_15_quant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: @*
dtype0�
Omodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_15_quant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_15_quant_dense_63_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

: @*
narrow_range(�
model_15/quant_dense_63/MatMulMatMulKmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
.model_15/quant_dense_63/BiasAdd/ReadVariableOpReadVariableOp7model_15_quant_dense_63_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_15/quant_dense_63/BiasAddBiasAdd(model_15/quant_dense_63/MatMul:product:06model_15/quant_dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Pmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_15_quant_dense_63_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_15_quant_dense_63_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars(model_15/quant_dense_63/BiasAdd:output:0Xmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityKmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�)
NoOpNoOp/^model_15/quant_dense_53/BiasAdd/ReadVariableOpN^model_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_54/BiasAdd/ReadVariableOpN^model_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_55/BiasAdd/ReadVariableOpN^model_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_56/BiasAdd/ReadVariableOpN^model_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_57/BiasAdd/ReadVariableOpN^model_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_58/BiasAdd/ReadVariableOpN^model_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_59/BiasAdd/ReadVariableOpN^model_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_60/BiasAdd/ReadVariableOpN^model_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_61/BiasAdd/ReadVariableOpN^model_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_62/BiasAdd/ReadVariableOpN^model_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1/^model_15/quant_dense_63/BiasAdd/ReadVariableOpN^model_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1T^model_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpV^model_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.model_15/quant_dense_53/BiasAdd/ReadVariableOp.model_15/quant_dense_53/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_53/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_53/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_54/BiasAdd/ReadVariableOp.model_15/quant_dense_54/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_54/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_54/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_55/BiasAdd/ReadVariableOp.model_15/quant_dense_55/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_55/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_55/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_56/BiasAdd/ReadVariableOp.model_15/quant_dense_56/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_56/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_56/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_57/BiasAdd/ReadVariableOp.model_15/quant_dense_57/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_57/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_57/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_58/BiasAdd/ReadVariableOp.model_15/quant_dense_58/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_58/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_58/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_59/BiasAdd/ReadVariableOp.model_15/quant_dense_59/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_59/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_59/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_60/BiasAdd/ReadVariableOp.model_15/quant_dense_60/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_60/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_60/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_61/BiasAdd/ReadVariableOp.model_15/quant_dense_61/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_61/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_61/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_62/BiasAdd/ReadVariableOp.model_15/quant_dense_62/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_62/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_62/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12`
.model_15/quant_dense_63/BiasAdd/ReadVariableOp.model_15/quant_dense_63/BiasAdd/ReadVariableOp2�
Mmodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_15/quant_dense_63/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_15/quant_dense_63/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Smodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpSmodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Umodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Umodel_15/quantize_layer_15/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_16
�
�
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3755215

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource: @J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:@K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

: @*
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

: @*
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
:��������� : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
0__inference_quant_dense_54_layer_call_fn_3754187

inputs
unknown:@2
	unknown_0: 
	unknown_1: 
	unknown_2:2
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
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3751738o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
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
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3755049

inputs=
+lastvaluequant_rank_readvariableop_resource: /
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource: @
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
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

: *
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: �
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

: *
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

: *
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

: *
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

: *
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� h
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
:��������� �
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
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
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_161
serving_default_input_16:0���������@B
quant_dense_630
StatefulPartitionedCall:0���������@tensorflow/serving/predict:��
�
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
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�
quantize_layer_15_min
quantize_layer_15_max
quantizer_vars
optimizer_step
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	layer
optimizer_step
_weight_vars

kernel_min
 
kernel_max
!_quantize_activations
"post_activation_min
#post_activation_max
$_output_quantizers
%	variables
&trainable_variables
'regularization_losses
(	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	)layer
*optimizer_step
+_weight_vars
,
kernel_min
-
kernel_max
._quantize_activations
/post_activation_min
0post_activation_max
1_output_quantizers
2	variables
3trainable_variables
4regularization_losses
5	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	6layer
7optimizer_step
8_weight_vars
9
kernel_min
:
kernel_max
;_quantize_activations
<post_activation_min
=post_activation_max
>_output_quantizers
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	Clayer
Doptimizer_step
E_weight_vars
F
kernel_min
G
kernel_max
H_quantize_activations
Ipost_activation_min
Jpost_activation_max
K_output_quantizers
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	Player
Qoptimizer_step
R_weight_vars
S
kernel_min
T
kernel_max
U_quantize_activations
Vpost_activation_min
Wpost_activation_max
X_output_quantizers
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	]layer
^optimizer_step
__weight_vars
`
kernel_min
a
kernel_max
b_quantize_activations
cpost_activation_min
dpost_activation_max
e_output_quantizers
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	jlayer
koptimizer_step
l_weight_vars
m
kernel_min
n
kernel_max
o_quantize_activations
ppost_activation_min
qpost_activation_max
r_output_quantizers
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	wlayer
xoptimizer_step
y_weight_vars
z
kernel_min
{
kernel_max
|_quantize_activations
}post_activation_min
~post_activation_max
_output_quantizers
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�post_activation_min
�post_activation_max
�_output_quantizers
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�post_activation_min
�post_activation_max
�_output_quantizers
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�post_activation_min
�post_activation_max
�_output_quantizers
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
0
1
2
�3
�4
5
6
 7
"8
#9
�10
�11
*12
,13
-14
/15
016
�17
�18
719
920
:21
<22
=23
�24
�25
D26
F27
G28
I29
J30
�31
�32
Q33
S34
T35
V36
W37
�38
�39
^40
`41
a42
c43
d44
�45
�46
k47
m48
n49
p50
q51
�52
�53
x54
z55
{56
}57
~58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
/:- 2'quantize_layer_15/quantize_layer_15_min
/:- 2'quantize_layer_15/quantize_layer_15_max
:
min_var
max_var"
trackable_dict_wrapper
(:& 2 quantize_layer_15/optimizer_step
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_53/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_53/kernel_min
!: 2quant_dense_53/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_53/post_activation_min
*:( 2"quant_dense_53/post_activation_max
 "
trackable_list_wrapper
S
�0
�1
2
3
 4
"5
#6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_54/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_54/kernel_min
!: 2quant_dense_54/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_54/post_activation_min
*:( 2"quant_dense_54/post_activation_max
 "
trackable_list_wrapper
S
�0
�1
*2
,3
-4
/5
06"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_55/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_55/kernel_min
!: 2quant_dense_55/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_55/post_activation_min
*:( 2"quant_dense_55/post_activation_max
 "
trackable_list_wrapper
S
�0
�1
72
93
:4
<5
=6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_56/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_56/kernel_min
!: 2quant_dense_56/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_56/post_activation_min
*:( 2"quant_dense_56/post_activation_max
 "
trackable_list_wrapper
S
�0
�1
D2
F3
G4
I5
J6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_57/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_57/kernel_min
!: 2quant_dense_57/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_57/post_activation_min
*:( 2"quant_dense_57/post_activation_max
 "
trackable_list_wrapper
S
�0
�1
Q2
S3
T4
V5
W6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_58/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_58/kernel_min
!: 2quant_dense_58/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_58/post_activation_min
*:( 2"quant_dense_58/post_activation_max
 "
trackable_list_wrapper
S
�0
�1
^2
`3
a4
c5
d6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_59/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_59/kernel_min
!: 2quant_dense_59/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_59/post_activation_min
*:( 2"quant_dense_59/post_activation_max
 "
trackable_list_wrapper
S
�0
�1
k2
m3
n4
p5
q6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_60/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_60/kernel_min
!: 2quant_dense_60/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_60/post_activation_min
*:( 2"quant_dense_60/post_activation_max
 "
trackable_list_wrapper
S
�0
�1
x2
z3
{4
}5
~6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_61/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_61/kernel_min
!: 2quant_dense_61/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_61/post_activation_min
*:( 2"quant_dense_61/post_activation_max
 "
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_62/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_62/kernel_min
!: 2quant_dense_62/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_62/post_activation_min
*:( 2"quant_dense_62/post_activation_max
 "
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_63/optimizer_step
(
�0"
trackable_list_wrapper
!: 2quant_dense_63/kernel_min
!: 2quant_dense_63/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_63/post_activation_min
*:( 2"quant_dense_63/post_activation_max
 "
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
!:@@2dense_53/kernel
:@2dense_53/bias
!:@22dense_54/kernel
:22dense_54/bias
!:2 2dense_55/kernel
: 2dense_55/bias
!: 2dense_56/kernel
:2dense_56/bias
!:2dense_57/kernel
:2dense_57/bias
!:2dense_58/kernel
:2dense_58/bias
!:2dense_59/kernel
:2dense_59/bias
!:2dense_60/kernel
:2dense_60/bias
!: 2dense_61/kernel
: 2dense_61/bias
!:  2dense_62/kernel
: 2dense_62/bias
!: @2dense_63/kernel
:@2dense_63/bias
�
0
1
2
3
4
 5
"6
#7
*8
,9
-10
/11
012
713
914
:15
<16
=17
D18
F19
G20
I21
J22
Q23
S24
T25
V26
W27
^28
`29
a30
c31
d32
k33
m34
n35
p36
q37
x38
z39
{40
}41
~42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
C
0
1
 2
"3
#4"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
C
*0
,1
-2
/3
04"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
C
70
91
:2
<3
=4"
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
C
D0
F1
G2
I3
J4"
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
C
Q0
S1
T2
V3
W4"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
C
^0
`1
a2
c3
d4"
trackable_list_wrapper
'
]0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
C
k0
m1
n2
p3
q4"
trackable_list_wrapper
'
j0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
C
x0
z1
{2
}3
~4"
trackable_list_wrapper
'
w0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1
�0
�2"
trackable_tuple_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
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
min_var
 max_var"
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
,min_var
-max_var"
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
9min_var
:max_var"
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
Fmin_var
Gmax_var"
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
Smin_var
Tmax_var"
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
`min_var
amax_var"
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
mmin_var
nmax_var"
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
zmin_var
{max_var"
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
<
�min_var
�max_var"
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
<
�min_var
�max_var"
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
<
�min_var
�max_var"
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
&:$@@2Adam/dense_53/kernel/m
 :@2Adam/dense_53/bias/m
&:$@22Adam/dense_54/kernel/m
 :22Adam/dense_54/bias/m
&:$2 2Adam/dense_55/kernel/m
 : 2Adam/dense_55/bias/m
&:$ 2Adam/dense_56/kernel/m
 :2Adam/dense_56/bias/m
&:$2Adam/dense_57/kernel/m
 :2Adam/dense_57/bias/m
&:$2Adam/dense_58/kernel/m
 :2Adam/dense_58/bias/m
&:$2Adam/dense_59/kernel/m
 :2Adam/dense_59/bias/m
&:$2Adam/dense_60/kernel/m
 :2Adam/dense_60/bias/m
&:$ 2Adam/dense_61/kernel/m
 : 2Adam/dense_61/bias/m
&:$  2Adam/dense_62/kernel/m
 : 2Adam/dense_62/bias/m
&:$ @2Adam/dense_63/kernel/m
 :@2Adam/dense_63/bias/m
&:$@@2Adam/dense_53/kernel/v
 :@2Adam/dense_53/bias/v
&:$@22Adam/dense_54/kernel/v
 :22Adam/dense_54/bias/v
&:$2 2Adam/dense_55/kernel/v
 : 2Adam/dense_55/bias/v
&:$ 2Adam/dense_56/kernel/v
 :2Adam/dense_56/bias/v
&:$2Adam/dense_57/kernel/v
 :2Adam/dense_57/bias/v
&:$2Adam/dense_58/kernel/v
 :2Adam/dense_58/bias/v
&:$2Adam/dense_59/kernel/v
 :2Adam/dense_59/bias/v
&:$2Adam/dense_60/kernel/v
 :2Adam/dense_60/bias/v
&:$ 2Adam/dense_61/kernel/v
 : 2Adam/dense_61/bias/v
&:$  2Adam/dense_62/kernel/v
 : 2Adam/dense_62/bias/v
&:$ @2Adam/dense_63/kernel/v
 :@2Adam/dense_63/bias/v
�2�
*__inference_model_15_layer_call_fn_3750834
*__inference_model_15_layer_call_fn_3753054
*__inference_model_15_layer_call_fn_3753195
*__inference_model_15_layer_call_fn_3752460�
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
E__inference_model_15_layer_call_and_return_conditional_losses_3753390
E__inference_model_15_layer_call_and_return_conditional_losses_3753993
E__inference_model_15_layer_call_and_return_conditional_losses_3752612
E__inference_model_15_layer_call_and_return_conditional_losses_3752764�
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
"__inference__wrapped_model_3750288input_16"�
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
3__inference_quantize_layer_15_layer_call_fn_3754002
3__inference_quantize_layer_15_layer_call_fn_3754011�
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
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3754020
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3754041�
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
0__inference_quant_dense_53_layer_call_fn_3754058
0__inference_quant_dense_53_layer_call_fn_3754075�
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
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3754096
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3754153�
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
0__inference_quant_dense_54_layer_call_fn_3754170
0__inference_quant_dense_54_layer_call_fn_3754187�
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
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3754208
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3754265�
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
0__inference_quant_dense_55_layer_call_fn_3754282
0__inference_quant_dense_55_layer_call_fn_3754299�
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
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3754320
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3754377�
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
0__inference_quant_dense_56_layer_call_fn_3754394
0__inference_quant_dense_56_layer_call_fn_3754411�
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
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3754432
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3754489�
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
0__inference_quant_dense_57_layer_call_fn_3754506
0__inference_quant_dense_57_layer_call_fn_3754523�
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
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3754544
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3754601�
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
0__inference_quant_dense_58_layer_call_fn_3754618
0__inference_quant_dense_58_layer_call_fn_3754635�
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
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3754656
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3754713�
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
0__inference_quant_dense_59_layer_call_fn_3754730
0__inference_quant_dense_59_layer_call_fn_3754747�
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
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3754768
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3754825�
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
0__inference_quant_dense_60_layer_call_fn_3754842
0__inference_quant_dense_60_layer_call_fn_3754859�
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
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3754880
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3754937�
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
0__inference_quant_dense_61_layer_call_fn_3754954
0__inference_quant_dense_61_layer_call_fn_3754971�
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
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3754992
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3755049�
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
0__inference_quant_dense_62_layer_call_fn_3755066
0__inference_quant_dense_62_layer_call_fn_3755083�
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
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3755104
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3755161�
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
0__inference_quant_dense_63_layer_call_fn_3755178
0__inference_quant_dense_63_layer_call_fn_3755195�
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
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3755215
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3755271�
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
%__inference_signature_wrapper_3752913input_16"�
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
 �
"__inference__wrapped_model_3750288�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������1�.
'�$
"�
input_16���������@
� "?�<
:
quant_dense_63(�%
quant_dense_63���������@�
E__inference_model_15_layer_call_and_return_conditional_losses_3752612�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������9�6
/�,
"�
input_16���������@
p 

 
� "%�"
�
0���������@
� �
E__inference_model_15_layer_call_and_return_conditional_losses_3752764�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������9�6
/�,
"�
input_16���������@
p

 
� "%�"
�
0���������@
� �
E__inference_model_15_layer_call_and_return_conditional_losses_3753390�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������7�4
-�*
 �
inputs���������@
p 

 
� "%�"
�
0���������@
� �
E__inference_model_15_layer_call_and_return_conditional_losses_3753993�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������7�4
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
*__inference_model_15_layer_call_fn_3750834�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������9�6
/�,
"�
input_16���������@
p 

 
� "����������@�
*__inference_model_15_layer_call_fn_3752460�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������9�6
/�,
"�
input_16���������@
p

 
� "����������@�
*__inference_model_15_layer_call_fn_3753054�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������7�4
-�*
 �
inputs���������@
p 

 
� "����������@�
*__inference_model_15_layer_call_fn_3753195�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������7�4
-�*
 �
inputs���������@
p

 
� "����������@�
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3754096f� �"#3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
K__inference_quant_dense_53_layer_call_and_return_conditional_losses_3754153f� �"#3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
0__inference_quant_dense_53_layer_call_fn_3754058Y� �"#3�0
)�&
 �
inputs���������@
p 
� "����������@�
0__inference_quant_dense_53_layer_call_fn_3754075Y� �"#3�0
)�&
 �
inputs���������@
p
� "����������@�
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3754208f�,-�/03�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������2
� �
K__inference_quant_dense_54_layer_call_and_return_conditional_losses_3754265f�,-�/03�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������2
� �
0__inference_quant_dense_54_layer_call_fn_3754170Y�,-�/03�0
)�&
 �
inputs���������@
p 
� "����������2�
0__inference_quant_dense_54_layer_call_fn_3754187Y�,-�/03�0
)�&
 �
inputs���������@
p
� "����������2�
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3754320f�9:�<=3�0
)�&
 �
inputs���������2
p 
� "%�"
�
0��������� 
� �
K__inference_quant_dense_55_layer_call_and_return_conditional_losses_3754377f�9:�<=3�0
)�&
 �
inputs���������2
p
� "%�"
�
0��������� 
� �
0__inference_quant_dense_55_layer_call_fn_3754282Y�9:�<=3�0
)�&
 �
inputs���������2
p 
� "���������� �
0__inference_quant_dense_55_layer_call_fn_3754299Y�9:�<=3�0
)�&
 �
inputs���������2
p
� "���������� �
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3754432f�FG�IJ3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0���������
� �
K__inference_quant_dense_56_layer_call_and_return_conditional_losses_3754489f�FG�IJ3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0���������
� �
0__inference_quant_dense_56_layer_call_fn_3754394Y�FG�IJ3�0
)�&
 �
inputs��������� 
p 
� "�����������
0__inference_quant_dense_56_layer_call_fn_3754411Y�FG�IJ3�0
)�&
 �
inputs��������� 
p
� "�����������
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3754544f�ST�VW3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
K__inference_quant_dense_57_layer_call_and_return_conditional_losses_3754601f�ST�VW3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
0__inference_quant_dense_57_layer_call_fn_3754506Y�ST�VW3�0
)�&
 �
inputs���������
p 
� "�����������
0__inference_quant_dense_57_layer_call_fn_3754523Y�ST�VW3�0
)�&
 �
inputs���������
p
� "�����������
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3754656f�`a�cd3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
K__inference_quant_dense_58_layer_call_and_return_conditional_losses_3754713f�`a�cd3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
0__inference_quant_dense_58_layer_call_fn_3754618Y�`a�cd3�0
)�&
 �
inputs���������
p 
� "�����������
0__inference_quant_dense_58_layer_call_fn_3754635Y�`a�cd3�0
)�&
 �
inputs���������
p
� "�����������
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3754768f�mn�pq3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
K__inference_quant_dense_59_layer_call_and_return_conditional_losses_3754825f�mn�pq3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
0__inference_quant_dense_59_layer_call_fn_3754730Y�mn�pq3�0
)�&
 �
inputs���������
p 
� "�����������
0__inference_quant_dense_59_layer_call_fn_3754747Y�mn�pq3�0
)�&
 �
inputs���������
p
� "�����������
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3754880f�z{�}~3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
K__inference_quant_dense_60_layer_call_and_return_conditional_losses_3754937f�z{�}~3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
0__inference_quant_dense_60_layer_call_fn_3754842Y�z{�}~3�0
)�&
 �
inputs���������
p 
� "�����������
0__inference_quant_dense_60_layer_call_fn_3754859Y�z{�}~3�0
)�&
 �
inputs���������
p
� "�����������
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3754992j������3�0
)�&
 �
inputs���������
p 
� "%�"
�
0��������� 
� �
K__inference_quant_dense_61_layer_call_and_return_conditional_losses_3755049j������3�0
)�&
 �
inputs���������
p
� "%�"
�
0��������� 
� �
0__inference_quant_dense_61_layer_call_fn_3754954]������3�0
)�&
 �
inputs���������
p 
� "���������� �
0__inference_quant_dense_61_layer_call_fn_3754971]������3�0
)�&
 �
inputs���������
p
� "���������� �
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3755104j������3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
K__inference_quant_dense_62_layer_call_and_return_conditional_losses_3755161j������3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
0__inference_quant_dense_62_layer_call_fn_3755066]������3�0
)�&
 �
inputs��������� 
p 
� "���������� �
0__inference_quant_dense_62_layer_call_fn_3755083]������3�0
)�&
 �
inputs��������� 
p
� "���������� �
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3755215j������3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0���������@
� �
K__inference_quant_dense_63_layer_call_and_return_conditional_losses_3755271j������3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0���������@
� �
0__inference_quant_dense_63_layer_call_fn_3755178]������3�0
)�&
 �
inputs��������� 
p 
� "����������@�
0__inference_quant_dense_63_layer_call_fn_3755195]������3�0
)�&
 �
inputs��������� 
p
� "����������@�
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3754020`3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
N__inference_quantize_layer_15_layer_call_and_return_conditional_losses_3754041`3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
3__inference_quantize_layer_15_layer_call_fn_3754002S3�0
)�&
 �
inputs���������@
p 
� "����������@�
3__inference_quantize_layer_15_layer_call_fn_3754011S3�0
)�&
 �
inputs���������@
p
� "����������@�
%__inference_signature_wrapper_3752913�f� �"#�,-�/0�9:�<=�FG�IJ�ST�VW�`a�cd�mn�pq�z{�}~������������������=�:
� 
3�0
.
input_16"�
input_16���������@"?�<
:
quant_dense_63(�%
quant_dense_63���������@