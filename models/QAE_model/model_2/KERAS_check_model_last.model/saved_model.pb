�7
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
2
Round
x"T
y"T"
Ttype:
2
	
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
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.12v2.11.0-94-ga3e2c692c188��3
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/classifier_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/v/classifier_out/bias
�
.Adam/v/classifier_out/bias/Read/ReadVariableOpReadVariableOpAdam/v/classifier_out/bias*
_output_shapes
:
*
dtype0
�
Adam/m/classifier_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/m/classifier_out/bias
�
.Adam/m/classifier_out/bias/Read/ReadVariableOpReadVariableOpAdam/m/classifier_out/bias*
_output_shapes
:
*
dtype0
�
Adam/v/classifier_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*-
shared_nameAdam/v/classifier_out/kernel
�
0Adam/v/classifier_out/kernel/Read/ReadVariableOpReadVariableOpAdam/v/classifier_out/kernel*
_output_shapes

:(
*
dtype0
�
Adam/m/classifier_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*-
shared_nameAdam/m/classifier_out/kernel
�
0Adam/m/classifier_out/kernel/Read/ReadVariableOpReadVariableOpAdam/m/classifier_out/kernel*
_output_shapes

:(
*
dtype0
�
Adam/v/fc5_class/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/v/fc5_class/bias
{
)Adam/v/fc5_class/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc5_class/bias*
_output_shapes
:(*
dtype0
�
Adam/m/fc5_class/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/m/fc5_class/bias
{
)Adam/m/fc5_class/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc5_class/bias*
_output_shapes
:(*
dtype0
�
Adam/v/fc5_class/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/v/fc5_class/kernel
�
+Adam/v/fc5_class/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc5_class/kernel*
_output_shapes

:(*
dtype0
�
Adam/m/fc5_class/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/m/fc5_class/kernel
�
+Adam/m/fc5_class/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc5_class/kernel*
_output_shapes

:(*
dtype0
�
/Adam/v/prune_low_magnitude_fc4_prunedclass/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/v/prune_low_magnitude_fc4_prunedclass/bias
�
CAdam/v/prune_low_magnitude_fc4_prunedclass/bias/Read/ReadVariableOpReadVariableOp/Adam/v/prune_low_magnitude_fc4_prunedclass/bias*
_output_shapes
:*
dtype0
�
/Adam/m/prune_low_magnitude_fc4_prunedclass/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/m/prune_low_magnitude_fc4_prunedclass/bias
�
CAdam/m/prune_low_magnitude_fc4_prunedclass/bias/Read/ReadVariableOpReadVariableOp/Adam/m/prune_low_magnitude_fc4_prunedclass/bias*
_output_shapes
:*
dtype0
�
1Adam/v/prune_low_magnitude_fc4_prunedclass/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/v/prune_low_magnitude_fc4_prunedclass/kernel
�
EAdam/v/prune_low_magnitude_fc4_prunedclass/kernel/Read/ReadVariableOpReadVariableOp1Adam/v/prune_low_magnitude_fc4_prunedclass/kernel*
_output_shapes

:*
dtype0
�
1Adam/m/prune_low_magnitude_fc4_prunedclass/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/m/prune_low_magnitude_fc4_prunedclass/kernel
�
EAdam/m/prune_low_magnitude_fc4_prunedclass/kernel/Read/ReadVariableOpReadVariableOp1Adam/m/prune_low_magnitude_fc4_prunedclass/kernel*
_output_shapes

:*
dtype0
�
Adam/v/encoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/v/encoder_output/bias
�
.Adam/v/encoder_output/bias/Read/ReadVariableOpReadVariableOpAdam/v/encoder_output/bias*
_output_shapes
:*
dtype0
�
Adam/m/encoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m/encoder_output/bias
�
.Adam/m/encoder_output/bias/Read/ReadVariableOpReadVariableOpAdam/m/encoder_output/bias*
_output_shapes
:*
dtype0
�
Adam/v/encoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/v/encoder_output/kernel
�
0Adam/v/encoder_output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encoder_output/kernel*
_output_shapes

:*
dtype0
�
Adam/m/encoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/m/encoder_output/kernel
�
0Adam/m/encoder_output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encoder_output/kernel*
_output_shapes

:*
dtype0
�
(Adam/v/prune_low_magnitude_fc3_prun/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/v/prune_low_magnitude_fc3_prun/bias
�
<Adam/v/prune_low_magnitude_fc3_prun/bias/Read/ReadVariableOpReadVariableOp(Adam/v/prune_low_magnitude_fc3_prun/bias*
_output_shapes
:*
dtype0
�
(Adam/m/prune_low_magnitude_fc3_prun/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/m/prune_low_magnitude_fc3_prun/bias
�
<Adam/m/prune_low_magnitude_fc3_prun/bias/Read/ReadVariableOpReadVariableOp(Adam/m/prune_low_magnitude_fc3_prun/bias*
_output_shapes
:*
dtype0
�
*Adam/v/prune_low_magnitude_fc3_prun/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/v/prune_low_magnitude_fc3_prun/kernel
�
>Adam/v/prune_low_magnitude_fc3_prun/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/prune_low_magnitude_fc3_prun/kernel*
_output_shapes

:*
dtype0
�
*Adam/m/prune_low_magnitude_fc3_prun/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/m/prune_low_magnitude_fc3_prun/kernel
�
>Adam/m/prune_low_magnitude_fc3_prun/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/prune_low_magnitude_fc3_prun/kernel*
_output_shapes

:*
dtype0
�
(Adam/v/prune_low_magnitude_fc2_prun/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/v/prune_low_magnitude_fc2_prun/bias
�
<Adam/v/prune_low_magnitude_fc2_prun/bias/Read/ReadVariableOpReadVariableOp(Adam/v/prune_low_magnitude_fc2_prun/bias*
_output_shapes
:*
dtype0
�
(Adam/m/prune_low_magnitude_fc2_prun/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/m/prune_low_magnitude_fc2_prun/bias
�
<Adam/m/prune_low_magnitude_fc2_prun/bias/Read/ReadVariableOpReadVariableOp(Adam/m/prune_low_magnitude_fc2_prun/bias*
_output_shapes
:*
dtype0
�
*Adam/v/prune_low_magnitude_fc2_prun/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/v/prune_low_magnitude_fc2_prun/kernel
�
>Adam/v/prune_low_magnitude_fc2_prun/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/prune_low_magnitude_fc2_prun/kernel*
_output_shapes

:*
dtype0
�
*Adam/m/prune_low_magnitude_fc2_prun/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/m/prune_low_magnitude_fc2_prun/kernel
�
>Adam/m/prune_low_magnitude_fc2_prun/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/prune_low_magnitude_fc2_prun/kernel*
_output_shapes

:*
dtype0
v
Adam/v/fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/v/fc1/bias
o
#Adam/v/fc1/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc1/bias*
_output_shapes
:*
dtype0
v
Adam/m/fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/m/fc1/bias
o
#Adam/m/fc1/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc1/bias*
_output_shapes
:*
dtype0
~
Adam/v/fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_nameAdam/v/fc1/kernel
w
%Adam/v/fc1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc1/kernel*
_output_shapes

:@*
dtype0
~
Adam/m/fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_nameAdam/m/fc1/kernel
w
%Adam/m/fc1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc1/kernel*
_output_shapes

:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
(prune_low_magnitude_fc4_prunedclass/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(prune_low_magnitude_fc4_prunedclass/bias
�
<prune_low_magnitude_fc4_prunedclass/bias/Read/ReadVariableOpReadVariableOp(prune_low_magnitude_fc4_prunedclass/bias*
_output_shapes
:*
dtype0
�
*prune_low_magnitude_fc4_prunedclass/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*prune_low_magnitude_fc4_prunedclass/kernel
�
>prune_low_magnitude_fc4_prunedclass/kernel/Read/ReadVariableOpReadVariableOp*prune_low_magnitude_fc4_prunedclass/kernel*
_output_shapes

:*
dtype0
�
!prune_low_magnitude_fc3_prun/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!prune_low_magnitude_fc3_prun/bias
�
5prune_low_magnitude_fc3_prun/bias/Read/ReadVariableOpReadVariableOp!prune_low_magnitude_fc3_prun/bias*
_output_shapes
:*
dtype0
�
#prune_low_magnitude_fc3_prun/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#prune_low_magnitude_fc3_prun/kernel
�
7prune_low_magnitude_fc3_prun/kernel/Read/ReadVariableOpReadVariableOp#prune_low_magnitude_fc3_prun/kernel*
_output_shapes

:*
dtype0
�
!prune_low_magnitude_fc2_prun/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!prune_low_magnitude_fc2_prun/bias
�
5prune_low_magnitude_fc2_prun/bias/Read/ReadVariableOpReadVariableOp!prune_low_magnitude_fc2_prun/bias*
_output_shapes
:*
dtype0
�
#prune_low_magnitude_fc2_prun/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#prune_low_magnitude_fc2_prun/kernel
�
7prune_low_magnitude_fc2_prun/kernel/Read/ReadVariableOpReadVariableOp#prune_low_magnitude_fc2_prun/kernel*
_output_shapes

:*
dtype0
~
classifier_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameclassifier_out/bias
w
'classifier_out/bias/Read/ReadVariableOpReadVariableOpclassifier_out/bias*
_output_shapes
:
*
dtype0
�
classifier_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*&
shared_nameclassifier_out/kernel

)classifier_out/kernel/Read/ReadVariableOpReadVariableOpclassifier_out/kernel*
_output_shapes

:(
*
dtype0
t
fc5_class/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namefc5_class/bias
m
"fc5_class/bias/Read/ReadVariableOpReadVariableOpfc5_class/bias*
_output_shapes
:(*
dtype0
|
fc5_class/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*!
shared_namefc5_class/kernel
u
$fc5_class/kernel/Read/ReadVariableOpReadVariableOpfc5_class/kernel*
_output_shapes

:(*
dtype0
�
0prune_low_magnitude_fc4_prunedclass/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *A
shared_name20prune_low_magnitude_fc4_prunedclass/pruning_step
�
Dprune_low_magnitude_fc4_prunedclass/pruning_step/Read/ReadVariableOpReadVariableOp0prune_low_magnitude_fc4_prunedclass/pruning_step*
_output_shapes
: *
dtype0	
�
-prune_low_magnitude_fc4_prunedclass/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-prune_low_magnitude_fc4_prunedclass/threshold
�
Aprune_low_magnitude_fc4_prunedclass/threshold/Read/ReadVariableOpReadVariableOp-prune_low_magnitude_fc4_prunedclass/threshold*
_output_shapes
: *
dtype0
�
(prune_low_magnitude_fc4_prunedclass/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(prune_low_magnitude_fc4_prunedclass/mask
�
<prune_low_magnitude_fc4_prunedclass/mask/Read/ReadVariableOpReadVariableOp(prune_low_magnitude_fc4_prunedclass/mask*
_output_shapes

:*
dtype0
~
encoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameencoder_output/bias
w
'encoder_output/bias/Read/ReadVariableOpReadVariableOpencoder_output/bias*
_output_shapes
:*
dtype0
�
encoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameencoder_output/kernel

)encoder_output/kernel/Read/ReadVariableOpReadVariableOpencoder_output/kernel*
_output_shapes

:*
dtype0
�
)prune_low_magnitude_fc3_prun/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *:
shared_name+)prune_low_magnitude_fc3_prun/pruning_step
�
=prune_low_magnitude_fc3_prun/pruning_step/Read/ReadVariableOpReadVariableOp)prune_low_magnitude_fc3_prun/pruning_step*
_output_shapes
: *
dtype0	
�
&prune_low_magnitude_fc3_prun/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&prune_low_magnitude_fc3_prun/threshold
�
:prune_low_magnitude_fc3_prun/threshold/Read/ReadVariableOpReadVariableOp&prune_low_magnitude_fc3_prun/threshold*
_output_shapes
: *
dtype0
�
!prune_low_magnitude_fc3_prun/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!prune_low_magnitude_fc3_prun/mask
�
5prune_low_magnitude_fc3_prun/mask/Read/ReadVariableOpReadVariableOp!prune_low_magnitude_fc3_prun/mask*
_output_shapes

:*
dtype0
�
)prune_low_magnitude_fc2_prun/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *:
shared_name+)prune_low_magnitude_fc2_prun/pruning_step
�
=prune_low_magnitude_fc2_prun/pruning_step/Read/ReadVariableOpReadVariableOp)prune_low_magnitude_fc2_prun/pruning_step*
_output_shapes
: *
dtype0	
�
&prune_low_magnitude_fc2_prun/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&prune_low_magnitude_fc2_prun/threshold
�
:prune_low_magnitude_fc2_prun/threshold/Read/ReadVariableOpReadVariableOp&prune_low_magnitude_fc2_prun/threshold*
_output_shapes
: *
dtype0
�
!prune_low_magnitude_fc2_prun/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!prune_low_magnitude_fc2_prun/mask
�
5prune_low_magnitude_fc2_prun/mask/Read/ReadVariableOpReadVariableOp!prune_low_magnitude_fc2_prun/mask*
_output_shapes

:*
dtype0
h
fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fc1/bias
a
fc1/bias/Read/ReadVariableOpReadVariableOpfc1/bias*
_output_shapes
:*
dtype0
p

fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name
fc1/kernel
i
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel*
_output_shapes

:@*
dtype0
�
serving_default_encoder_inputPlaceholder*'
_output_shapes
:���������@*
dtype0*
shape:���������@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_input
fc1/kernelfc1/bias#prune_low_magnitude_fc2_prun/kernel!prune_low_magnitude_fc2_prun/mask!prune_low_magnitude_fc2_prun/bias#prune_low_magnitude_fc3_prun/kernel!prune_low_magnitude_fc3_prun/mask!prune_low_magnitude_fc3_prun/biasencoder_output/kernelencoder_output/bias*prune_low_magnitude_fc4_prunedclass/kernel(prune_low_magnitude_fc4_prunedclass/mask(prune_low_magnitude_fc4_prunedclass/biasfc5_class/kernelfc5_class/biasclassifier_out/kernelclassifier_out/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1596849

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
valueؔBԔ B̔
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
kernel_quantizer
bias_quantizer
kernel_quantizer_internal
bias_quantizer_internal
 
quantizers

!kernel
"bias*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)
activation
)	quantizer* 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0pruning_vars
	1layer
2prunable_weights
3mask
4	threshold
5pruning_step*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<
activation
<	quantizer* 
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cpruning_vars
	Dlayer
Eprunable_weights
Fmask
G	threshold
Hpruning_step*
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O
activation
O	quantizer* 
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias*
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^pruning_vars
	_layer
`prunable_weights
amask
b	threshold
cpruning_step*
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j
activation
j	quantizer* 
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qkernel_quantizer
rbias_quantizer
qkernel_quantizer_internal
rbias_quantizer_internal
s
quantizers

tkernel
ubias*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|
activation
|	quantizer* 
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�bias_quantizer
�kernel_quantizer_internal
�bias_quantizer_internal
�
quantizers
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
!0
"1
�2
�3
34
45
56
�7
�8
F9
G10
H11
V12
W13
�14
�15
a16
b17
c18
t19
u20
�21
�22*
r
!0
"1
�2
�3
�4
�5
V6
W7
�8
�9
t10
u11
�12
�13*

�0
�1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

!0
"1*

!0
"1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

0
1* 
ZT
VARIABLE_VALUE
fc1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfc1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
)
�0
�1
32
43
54*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�0*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�bias_quantizer
�kernel_quantizer_internal
�bias_quantizer_internal
�
quantizers
�kernel
	�bias*

�0*
oi
VARIABLE_VALUE!prune_low_magnitude_fc2_prun/mask4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE&prune_low_magnitude_fc2_prun/threshold9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE)prune_low_magnitude_fc2_prun/pruning_step<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
)
�0
�1
F2
G3
H4*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�0*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�bias_quantizer
�kernel_quantizer_internal
�bias_quantizer_internal
�
quantizers
�kernel
	�bias*

�0*
oi
VARIABLE_VALUE!prune_low_magnitude_fc3_prun/mask4layer_with_weights-2/mask/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE&prune_low_magnitude_fc3_prun/threshold9layer_with_weights-2/threshold/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE)prune_low_magnitude_fc3_prun/pruning_step<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

V0
W1*

V0
W1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEencoder_output/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEencoder_output/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
)
�0
�1
a2
b3
c4*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�0*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�bias_quantizer
�kernel_quantizer_internal
�bias_quantizer_internal
�
quantizers
�kernel
	�bias*

�0*
vp
VARIABLE_VALUE(prune_low_magnitude_fc4_prunedclass/mask4layer_with_weights-4/mask/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE-prune_low_magnitude_fc4_prunedclass/threshold9layer_with_weights-4/threshold/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE0prune_low_magnitude_fc4_prunedclass/pruning_step<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

t0
u1*

t0
u1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

q0
r1* 
`Z
VARIABLE_VALUEfc5_class/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEfc5_class/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

�0
�1* 
e_
VARIABLE_VALUEclassifier_out/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEclassifier_out/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUE#prune_low_magnitude_fc2_prun/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!prune_low_magnitude_fc2_prun/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#prune_low_magnitude_fc3_prun/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!prune_low_magnitude_fc3_prun/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*prune_low_magnitude_fc4_prunedclass/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(prune_low_magnitude_fc4_prunedclass/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 
C
30
41
52
F3
G4
H5
a6
b7
c8*
j
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
13*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

30
41
52*

10*
* 
* 
* 
* 
* 
* 
* 

�0
31
42*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 
* 

F0
G1
H2*

D0*
* 
* 
* 
* 
* 
* 
* 

�0
F1
G2*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�trace_0* 

a0
b1
c2*

_0*
* 
* 
* 
* 
* 
* 
* 

�0
a1
b2*

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
\V
VARIABLE_VALUEAdam/m/fc1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/fc1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/fc1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/fc1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/prune_low_magnitude_fc2_prun/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/prune_low_magnitude_fc2_prun/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/prune_low_magnitude_fc2_prun/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/prune_low_magnitude_fc2_prun/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/prune_low_magnitude_fc3_prun/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/prune_low_magnitude_fc3_prun/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/prune_low_magnitude_fc3_prun/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/prune_low_magnitude_fc3_prun/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/encoder_output/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/encoder_output/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/encoder_output/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/encoder_output/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/m/prune_low_magnitude_fc4_prunedclass/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/v/prune_low_magnitude_fc4_prunedclass/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE/Adam/m/prune_low_magnitude_fc4_prunedclass/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE/Adam/v/prune_low_magnitude_fc4_prunedclass/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/fc5_class/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/fc5_class/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/fc5_class/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/fc5_class/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/classifier_out/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/classifier_out/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/classifier_out/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/classifier_out/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamefc1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOp5prune_low_magnitude_fc2_prun/mask/Read/ReadVariableOp:prune_low_magnitude_fc2_prun/threshold/Read/ReadVariableOp=prune_low_magnitude_fc2_prun/pruning_step/Read/ReadVariableOp5prune_low_magnitude_fc3_prun/mask/Read/ReadVariableOp:prune_low_magnitude_fc3_prun/threshold/Read/ReadVariableOp=prune_low_magnitude_fc3_prun/pruning_step/Read/ReadVariableOp)encoder_output/kernel/Read/ReadVariableOp'encoder_output/bias/Read/ReadVariableOp<prune_low_magnitude_fc4_prunedclass/mask/Read/ReadVariableOpAprune_low_magnitude_fc4_prunedclass/threshold/Read/ReadVariableOpDprune_low_magnitude_fc4_prunedclass/pruning_step/Read/ReadVariableOp$fc5_class/kernel/Read/ReadVariableOp"fc5_class/bias/Read/ReadVariableOp)classifier_out/kernel/Read/ReadVariableOp'classifier_out/bias/Read/ReadVariableOp7prune_low_magnitude_fc2_prun/kernel/Read/ReadVariableOp5prune_low_magnitude_fc2_prun/bias/Read/ReadVariableOp7prune_low_magnitude_fc3_prun/kernel/Read/ReadVariableOp5prune_low_magnitude_fc3_prun/bias/Read/ReadVariableOp>prune_low_magnitude_fc4_prunedclass/kernel/Read/ReadVariableOp<prune_low_magnitude_fc4_prunedclass/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp%Adam/m/fc1/kernel/Read/ReadVariableOp%Adam/v/fc1/kernel/Read/ReadVariableOp#Adam/m/fc1/bias/Read/ReadVariableOp#Adam/v/fc1/bias/Read/ReadVariableOp>Adam/m/prune_low_magnitude_fc2_prun/kernel/Read/ReadVariableOp>Adam/v/prune_low_magnitude_fc2_prun/kernel/Read/ReadVariableOp<Adam/m/prune_low_magnitude_fc2_prun/bias/Read/ReadVariableOp<Adam/v/prune_low_magnitude_fc2_prun/bias/Read/ReadVariableOp>Adam/m/prune_low_magnitude_fc3_prun/kernel/Read/ReadVariableOp>Adam/v/prune_low_magnitude_fc3_prun/kernel/Read/ReadVariableOp<Adam/m/prune_low_magnitude_fc3_prun/bias/Read/ReadVariableOp<Adam/v/prune_low_magnitude_fc3_prun/bias/Read/ReadVariableOp0Adam/m/encoder_output/kernel/Read/ReadVariableOp0Adam/v/encoder_output/kernel/Read/ReadVariableOp.Adam/m/encoder_output/bias/Read/ReadVariableOp.Adam/v/encoder_output/bias/Read/ReadVariableOpEAdam/m/prune_low_magnitude_fc4_prunedclass/kernel/Read/ReadVariableOpEAdam/v/prune_low_magnitude_fc4_prunedclass/kernel/Read/ReadVariableOpCAdam/m/prune_low_magnitude_fc4_prunedclass/bias/Read/ReadVariableOpCAdam/v/prune_low_magnitude_fc4_prunedclass/bias/Read/ReadVariableOp+Adam/m/fc5_class/kernel/Read/ReadVariableOp+Adam/v/fc5_class/kernel/Read/ReadVariableOp)Adam/m/fc5_class/bias/Read/ReadVariableOp)Adam/v/fc5_class/bias/Read/ReadVariableOp0Adam/m/classifier_out/kernel/Read/ReadVariableOp0Adam/v/classifier_out/kernel/Read/ReadVariableOp.Adam/m/classifier_out/bias/Read/ReadVariableOp.Adam/v/classifier_out/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*F
Tin?
=2;				*
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
 __inference__traced_save_1600337
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
fc1/kernelfc1/bias!prune_low_magnitude_fc2_prun/mask&prune_low_magnitude_fc2_prun/threshold)prune_low_magnitude_fc2_prun/pruning_step!prune_low_magnitude_fc3_prun/mask&prune_low_magnitude_fc3_prun/threshold)prune_low_magnitude_fc3_prun/pruning_stepencoder_output/kernelencoder_output/bias(prune_low_magnitude_fc4_prunedclass/mask-prune_low_magnitude_fc4_prunedclass/threshold0prune_low_magnitude_fc4_prunedclass/pruning_stepfc5_class/kernelfc5_class/biasclassifier_out/kernelclassifier_out/bias#prune_low_magnitude_fc2_prun/kernel!prune_low_magnitude_fc2_prun/bias#prune_low_magnitude_fc3_prun/kernel!prune_low_magnitude_fc3_prun/bias*prune_low_magnitude_fc4_prunedclass/kernel(prune_low_magnitude_fc4_prunedclass/bias	iterationlearning_rateAdam/m/fc1/kernelAdam/v/fc1/kernelAdam/m/fc1/biasAdam/v/fc1/bias*Adam/m/prune_low_magnitude_fc2_prun/kernel*Adam/v/prune_low_magnitude_fc2_prun/kernel(Adam/m/prune_low_magnitude_fc2_prun/bias(Adam/v/prune_low_magnitude_fc2_prun/bias*Adam/m/prune_low_magnitude_fc3_prun/kernel*Adam/v/prune_low_magnitude_fc3_prun/kernel(Adam/m/prune_low_magnitude_fc3_prun/bias(Adam/v/prune_low_magnitude_fc3_prun/biasAdam/m/encoder_output/kernelAdam/v/encoder_output/kernelAdam/m/encoder_output/biasAdam/v/encoder_output/bias1Adam/m/prune_low_magnitude_fc4_prunedclass/kernel1Adam/v/prune_low_magnitude_fc4_prunedclass/kernel/Adam/m/prune_low_magnitude_fc4_prunedclass/bias/Adam/v/prune_low_magnitude_fc4_prunedclass/biasAdam/m/fc5_class/kernelAdam/v/fc5_class/kernelAdam/m/fc5_class/biasAdam/v/fc5_class/biasAdam/m/classifier_out/kernelAdam/v/classifier_out/kernelAdam/m/classifier_out/biasAdam/v/classifier_out/biastotal_1count_1totalcount*E
Tin>
<2:*
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
#__inference__traced_restore_1600518�1
�z
�
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1599820

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: '
readvariableop_4_resource:
identity��AssignVariableOp�AssignVariableOp_1�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: �
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(7
updateNoOp^AssignVariableOp*
_output_shapes
 �
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	X
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: [
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: �
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: �
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *H
else_branch9R7
5assert_greater_equal_Assert_AssertGuard_false_1599632*
output_shapes
: *G
then_branch8R6
4assert_greater_equal_Assert_AssertGuard_true_1599631�
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�{
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�No
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: |
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N{
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : O
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: G
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: Q

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: �
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	{
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�W
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RdS
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: |
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R O
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: M
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: }
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��L?�
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*%
else_branchR
cond_false_1599672*
output_shapes
: *$
then_branchR
cond_true_1599671q
cond/IdentityIdentitycond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: i
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 _
Mul/ReadVariableOpReadVariableOpcond_input_1*
_output_shapes

:*
dtype0h
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: r
ReadVariableOp_1ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ba
mul_1MulReadVariableOp_1:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_1AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_1:z:0*
T0*
_output_shapes

:[
add_2AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value/MinimumMinimum	add_2:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:r
ReadVariableOp_2ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_2:value:0*
T0*
_output_shapes

:M
add_3AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:r
ReadVariableOp_3ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0j
add_4AddV2ReadVariableOp_3:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_4:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_5MulReadVariableOp_4:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_5AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_5:z:0*
T0*
_output_shapes
:[
add_6AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_5:value:0*
T0*
_output_shapes
:I
add_7AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_7:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0f
add_8AddV2ReadVariableOp_6:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_8:z:0*
T0*'
_output_shapes
:����������
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^condJ^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond2�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
5assert_greater_equal_Assert_AssertGuard_false_1599632K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
��.assert_greater_equal/Assert/AssertGuard/Assert�
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp/^assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
K__inference_encoder_output_layer_call_and_return_conditional_losses_1599508

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�n
�
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1599434

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: '
readvariableop_4_resource:
identity��AssignVariableOp�AssignVariableOp_1�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: �
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(7
updateNoOp^AssignVariableOp*
_output_shapes
 �
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	X
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: [
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: �
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: �
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *H
else_branch9R7
5assert_greater_equal_Assert_AssertGuard_false_1599252*
output_shapes
: *G
then_branch8R6
4assert_greater_equal_Assert_AssertGuard_true_1599251�
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�{
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�No
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: |
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N{
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : O
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: G
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: Q

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: �
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	{
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�W
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RdS
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: |
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R O
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: M
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: }
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��L?�
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*%
else_branchR
cond_false_1599292*
output_shapes
: *$
then_branchR
cond_true_1599291q
cond/IdentityIdentitycond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: i
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 _
Mul/ReadVariableOpReadVariableOpcond_input_1*
_output_shapes

:*
dtype0h
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: r
ReadVariableOp_1ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aa
mul_1MulReadVariableOp_1:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_1AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_1:z:0*
T0*
_output_shapes

:[
add_2AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_2:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:r
ReadVariableOp_2ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_2:value:0*
T0*
_output_shapes

:M
add_3AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:r
ReadVariableOp_3ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0j
add_4AddV2ReadVariableOp_3:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_4:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_5MulReadVariableOp_4:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_5AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_5:z:0*
T0*
_output_shapes
:[
add_6AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_5:value:0*
T0*
_output_shapes
:I
add_7AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_7:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0f
add_8AddV2ReadVariableOp_6:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_8:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
d
H__inference_class_relu5_layer_call_and_return_conditional_losses_1595502

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������(F
ReluReluinputs*
T0*'
_output_shapes
:���������(E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������(D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������(r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������(P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������([
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������(I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������(M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������(R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������(W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������(d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������([
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������(P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������(]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������(Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������(V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������(L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������([
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������(l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������(Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������(:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�:
�
K__inference_classifier_out_layer_call_and_return_conditional_losses_1600100

inputs)
readvariableop_resource:(
'
readvariableop_3_resource:

identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�4classifier_out/kernel/Regularizer/Abs/ReadVariableOpG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(
*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:(
N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:(
@
NegNegtruediv:z:0*
T0*
_output_shapes

:(
D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:(
I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:(
N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:(
[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:(
\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Fv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(
T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:(
R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:(
P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:(
L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:(
h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:(
*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:(
M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:(
L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:(
R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:(
h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:(
*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:(
U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������
I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:
*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:
P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:
@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:
D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:
K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:
N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:
[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:
^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Fv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:
V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:
R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:
P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   GZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:
L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:
f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:
*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:
I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:
L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:
N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:
f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:
*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:
a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������
�
4classifier_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(
*
dtype0�
%classifier_out/kernel/Regularizer/AbsAbs<classifier_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(
x
'classifier_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%classifier_out/kernel/Regularizer/SumSum)classifier_out/kernel/Regularizer/Abs:y:00classifier_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'classifier_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%classifier_out/kernel/Regularizer/mulMul0classifier_out/kernel/Regularizer/mul/x:output:0.classifier_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_55^classifier_out/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52l
4classifier_out/kernel/Regularizer/Abs/ReadVariableOp4classifier_out/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
� 
^
B__inference_relu1_layer_call_and_return_conditional_losses_1594955

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/prune_low_magnitude_fc2_prun_cond_false_15977771
-prune_low_magnitude_fc2_prun_cond_placeholder3
/prune_low_magnitude_fc2_prun_cond_placeholder_13
/prune_low_magnitude_fc2_prun_cond_placeholder_23
/prune_low_magnitude_fc2_prun_cond_placeholder_3X
Tprune_low_magnitude_fc2_prun_cond_identity_prune_low_magnitude_fc2_prun_logicaland_1
0
,prune_low_magnitude_fc2_prun_cond_identity_1
l
&prune_low_magnitude_fc2_prun/cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
*prune_low_magnitude_fc2_prun/cond/IdentityIdentityTprune_low_magnitude_fc2_prun_cond_identity_prune_low_magnitude_fc2_prun_logicaland_1'^prune_low_magnitude_fc2_prun/cond/NoOp*
T0
*
_output_shapes
: �
,prune_low_magnitude_fc2_prun/cond/Identity_1Identity3prune_low_magnitude_fc2_prun/cond/Identity:output:0*
T0
*
_output_shapes
: "e
,prune_low_magnitude_fc2_prun_cond_identity_15prune_low_magnitude_fc2_prun/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
� 
�
Yprune_low_magnitude_fc4_prunedclass_assert_greater_equal_Assert_AssertGuard_false_1598230�
�prune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc4_prunedclass_assert_greater_equal_all
�
�prune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc4_prunedclass_assert_greater_equal_readvariableop	�
�prune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc4_prunedclass_assert_greater_equal_y	Z
Vprune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_identity_1
��Rprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert�
Yprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Yprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Yprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*a
valueXBV BPx (prune_low_magnitude_fc4_prunedclass/assert_greater_equal/ReadVariableOp:0) = �
Yprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*T
valueKBI BCy (prune_low_magnitude_fc4_prunedclass/assert_greater_equal/y:0) = �
Rprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/AssertAssert�prune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc4_prunedclass_assert_greater_equal_allbprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0bprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0bprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0�prune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc4_prunedclass_assert_greater_equal_readvariableopbprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0�prune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc4_prunedclass_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Tprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc4_prunedclass_assert_greater_equal_allS^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
Vprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity_1Identity]prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity:output:0Q^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Pprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/NoOpNoOpS^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "�
Vprune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_identity_1_prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2�
Rprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/AssertRprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
5assert_greater_equal_Assert_AssertGuard_false_1596205K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
��.assert_greater_equal/Assert/AssertGuard/Assert�
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp/^assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
� 
^
B__inference_relu1_layer_call_and_return_conditional_losses_1598780

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
cond_false_1595759
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
O
	cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 b
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: T
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
�Z
�

G__inference_classifier_layer_call_and_return_conditional_losses_1595610

inputs
fc1_1594901:@
fc1_1594903:6
$prune_low_magnitude_fc2_prun_1595030:6
$prune_low_magnitude_fc2_prun_1595032:2
$prune_low_magnitude_fc2_prun_1595034:6
$prune_low_magnitude_fc3_prun_1595161:6
$prune_low_magnitude_fc3_prun_1595163:2
$prune_low_magnitude_fc3_prun_1595165:(
encoder_output_1595231:$
encoder_output_1595233:=
+prune_low_magnitude_fc4_prunedclass_1595315:=
+prune_low_magnitude_fc4_prunedclass_1595317:9
+prune_low_magnitude_fc4_prunedclass_1595319:#
fc5_class_1595448:(
fc5_class_1595450:((
classifier_out_1595579:(
$
classifier_out_1595581:

identity��&classifier_out/StatefulPartitionedCall�4classifier_out/kernel/Regularizer/Abs/ReadVariableOp�&encoder_output/StatefulPartitionedCall�fc1/StatefulPartitionedCall�!fc5_class/StatefulPartitionedCall�/fc5_class/kernel/Regularizer/Abs/ReadVariableOp�4prune_low_magnitude_fc2_prun/StatefulPartitionedCall�4prune_low_magnitude_fc3_prun/StatefulPartitionedCall�;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp�
fc1/StatefulPartitionedCallStatefulPartitionedCallinputsfc1_1594901fc1_1594903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_1594900�
relu1/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_1594955�
4prune_low_magnitude_fc2_prun/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0$prune_low_magnitude_fc2_prun_1595030$prune_low_magnitude_fc2_prun_1595032$prune_low_magnitude_fc2_prun_1595034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1595029�
relu2/PartitionedCallPartitionedCall=prune_low_magnitude_fc2_prun/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_1595086�
4prune_low_magnitude_fc3_prun/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0$prune_low_magnitude_fc3_prun_1595161$prune_low_magnitude_fc3_prun_1595163$prune_low_magnitude_fc3_prun_1595165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1595160�
relu3_enc/PartitionedCallPartitionedCall=prune_low_magnitude_fc3_prun/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1595217�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall"relu3_enc/PartitionedCall:output:0encoder_output_1595231encoder_output_1595233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_output_layer_call_and_return_conditional_losses_1595230�
;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0+prune_low_magnitude_fc4_prunedclass_1595315+prune_low_magnitude_fc4_prunedclass_1595317+prune_low_magnitude_fc4_prunedclass_1595319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *i
fdRb
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1595314�
prunclass_relu4/PartitionedCallPartitionedCallDprune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1595371�
!fc5_class/StatefulPartitionedCallStatefulPartitionedCall(prunclass_relu4/PartitionedCall:output:0fc5_class_1595448fc5_class_1595450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_fc5_class_layer_call_and_return_conditional_losses_1595447�
class_relu5/PartitionedCallPartitionedCall*fc5_class/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_class_relu5_layer_call_and_return_conditional_losses_1595502�
&classifier_out/StatefulPartitionedCallStatefulPartitionedCall$class_relu5/PartitionedCall:output:0classifier_out_1595579classifier_out_1595581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_classifier_out_layer_call_and_return_conditional_losses_1595578�
!classifier_output/PartitionedCallPartitionedCall/classifier_out/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_classifier_output_layer_call_and_return_conditional_losses_1595589�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+prune_low_magnitude_fc4_prunedclass_1595315<^prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
/fc5_class/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfc5_class_1595448*
_output_shapes

:(*
dtype0�
 fc5_class/kernel/Regularizer/AbsAbs7fc5_class/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(s
"fc5_class/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 fc5_class/kernel/Regularizer/SumSum$fc5_class/kernel/Regularizer/Abs:y:0+fc5_class/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"fc5_class/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
 fc5_class/kernel/Regularizer/mulMul+fc5_class/kernel/Regularizer/mul/x:output:0)fc5_class/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4classifier_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpclassifier_out_1595579*
_output_shapes

:(
*
dtype0�
%classifier_out/kernel/Regularizer/AbsAbs<classifier_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(
x
'classifier_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%classifier_out/kernel/Regularizer/SumSum)classifier_out/kernel/Regularizer/Abs:y:00classifier_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'classifier_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%classifier_out/kernel/Regularizer/mulMul0classifier_out/kernel/Regularizer/mul/x:output:0.classifier_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*classifier_output/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp'^classifier_out/StatefulPartitionedCall5^classifier_out/kernel/Regularizer/Abs/ReadVariableOp'^encoder_output/StatefulPartitionedCall^fc1/StatefulPartitionedCall"^fc5_class/StatefulPartitionedCall0^fc5_class/kernel/Regularizer/Abs/ReadVariableOp5^prune_low_magnitude_fc2_prun/StatefulPartitionedCall5^prune_low_magnitude_fc3_prun/StatefulPartitionedCall<^prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCallJ^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������@: : : : : : : : : : : : : : : : : 2P
&classifier_out/StatefulPartitionedCall&classifier_out/StatefulPartitionedCall2l
4classifier_out/kernel/Regularizer/Abs/ReadVariableOp4classifier_out/kernel/Regularizer/Abs/ReadVariableOp2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2F
!fc5_class/StatefulPartitionedCall!fc5_class/StatefulPartitionedCall2b
/fc5_class/kernel/Regularizer/Abs/ReadVariableOp/fc5_class/kernel/Regularizer/Abs/ReadVariableOp2l
4prune_low_magnitude_fc2_prun/StatefulPartitionedCall4prune_low_magnitude_fc2_prun/StatefulPartitionedCall2l
4prune_low_magnitude_fc3_prun/StatefulPartitionedCall4prune_low_magnitude_fc3_prun/StatefulPartitionedCall2z
;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall2�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�>
�
cond_true_15960093
)cond_greaterequal_readvariableop_resource:	 >
,cond_pruning_ops_abs_readvariableop_resource:0
cond_assignvariableop_resource:*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
��cond/AssignVariableOp�cond/AssignVariableOp_1� cond/GreaterEqual/ReadVariableOp�cond/LessEqual/ReadVariableOp�cond/Sub/ReadVariableOp�#cond/pruning_ops/Abs/ReadVariableOp�
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	V
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	S
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N~
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: N
cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�NM
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : ^
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: V
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: `
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: y
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	M

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�f
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: Q
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rdb
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: N
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ^

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: \
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: O

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0q
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:X
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :�m
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: [
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?r
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: q
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: Z
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: _
cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
cond/pruning_ops/MaximumMaximumcond/pruning_ops/Round:y:0#cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: m
cond/pruning_ops/Cast_1Castcond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: q
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:�Z
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :��
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:�:�Z
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: `
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: Z
cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_2Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: b
 cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2_1GatherV2!cond/pruning_ops/TopKV2:indices:0cond/pruning_ops/sub_2:z:0)cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:Z
cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value
B :�`
cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zb
 cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
cond/pruning_ops/one_hotOneHot$cond/pruning_ops/GatherV2_1:output:0 cond/pruning_ops/Size_2:output:0'cond/pruning_ops/one_hot/Const:output:0)cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes	
:�q
 cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
cond/pruning_ops/Reshape_1Reshape!cond/pruning_ops/one_hot:output:0)cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
cond/pruning_ops/LogicalOr	LogicalOr!cond/pruning_ops/GreaterEqual:z:0#cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:i
	cond/CastCastcond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: `
cond/Identity_1Identitycond/Identity:output:0
^cond/NoOp*
T0
*
_output_shapes
: �
	cond/NoOpNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�	
�
6prune_low_magnitude_fc4_prunedclass_cond_false_15982708
4prune_low_magnitude_fc4_prunedclass_cond_placeholder:
6prune_low_magnitude_fc4_prunedclass_cond_placeholder_1:
6prune_low_magnitude_fc4_prunedclass_cond_placeholder_2:
6prune_low_magnitude_fc4_prunedclass_cond_placeholder_3f
bprune_low_magnitude_fc4_prunedclass_cond_identity_prune_low_magnitude_fc4_prunedclass_logicaland_1
7
3prune_low_magnitude_fc4_prunedclass_cond_identity_1
s
-prune_low_magnitude_fc4_prunedclass/cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
1prune_low_magnitude_fc4_prunedclass/cond/IdentityIdentitybprune_low_magnitude_fc4_prunedclass_cond_identity_prune_low_magnitude_fc4_prunedclass_logicaland_1.^prune_low_magnitude_fc4_prunedclass/cond/NoOp*
T0
*
_output_shapes
: �
3prune_low_magnitude_fc4_prunedclass/cond/Identity_1Identity:prune_low_magnitude_fc4_prunedclass/cond/Identity:output:0*
T0
*
_output_shapes
: "s
3prune_low_magnitude_fc4_prunedclass_cond_identity_1<prune_low_magnitude_fc4_prunedclass/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
�
j
N__inference_classifier_output_layer_call_and_return_conditional_losses_1600110

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
,__inference_classifier_layer_call_fn_1596638
encoder_input
unknown:@
	unknown_0:
	unknown_1:	 
	unknown_2:
	unknown_3:
	unknown_4: 
	unknown_5:
	unknown_6:	 
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10:

unknown_11:

unknown_12:

unknown_13:	 

unknown_14:

unknown_15:

unknown_16: 

unknown_17:

unknown_18:(

unknown_19:(

unknown_20:(


unknown_21:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*-
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_classifier_layer_call_and_return_conditional_losses_1596538o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������@: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�>
�
cond_true_15957583
)cond_greaterequal_readvariableop_resource:	 >
,cond_pruning_ops_abs_readvariableop_resource:0
cond_assignvariableop_resource:*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
��cond/AssignVariableOp�cond/AssignVariableOp_1� cond/GreaterEqual/ReadVariableOp�cond/LessEqual/ReadVariableOp�cond/Sub/ReadVariableOp�#cond/pruning_ops/Abs/ReadVariableOp�
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	V
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	S
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N~
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: N
cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�NM
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : ^
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: V
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: `
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: y
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	M

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�f
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: Q
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rdb
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: N
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ^

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: \
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: O

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0q
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:W
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value	B : m
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: [
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?r
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: q
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: Z
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: _
cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
cond/pruning_ops/MaximumMaximumcond/pruning_ops/Round:y:0#cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: m
cond/pruning_ops/Cast_1Castcond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: q
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes
: Y
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0* 
_output_shapes
: : Z
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: `
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: Z
cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_2Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: b
 cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2_1GatherV2!cond/pruning_ops/TopKV2:indices:0cond/pruning_ops/sub_2:z:0)cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:Y
cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value	B : `
cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zb
 cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
cond/pruning_ops/one_hotOneHot$cond/pruning_ops/GatherV2_1:output:0 cond/pruning_ops/Size_2:output:0'cond/pruning_ops/one_hot/Const:output:0)cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes
: q
 cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
cond/pruning_ops/Reshape_1Reshape!cond/pruning_ops/one_hot:output:0)cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
cond/pruning_ops/LogicalOr	LogicalOr!cond/pruning_ops/GreaterEqual:z:0#cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:i
	cond/CastCastcond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: `
cond/Identity_1Identitycond/Identity:output:0
^cond/NoOp*
T0
*
_output_shapes
: �
	cond/NoOpNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
� 
d
H__inference_class_relu5_layer_call_and_return_conditional_losses_1600017

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������(F
ReluReluinputs*
T0*'
_output_shapes
:���������(E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������(D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������(r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������(P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������([
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������(I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������(M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������(R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������(W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������(d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������([
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������(P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������(]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������(Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������(V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������(L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������([
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������(l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������(Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������(:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
5assert_greater_equal_Assert_AssertGuard_false_1595719K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
��.assert_greater_equal/Assert/AssertGuard/Assert�
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp/^assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
Rprune_low_magnitude_fc2_prun_assert_greater_equal_Assert_AssertGuard_false_1597737�
�prune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_prun_assert_greater_equal_all
�
�prune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_prun_assert_greater_equal_readvariableop	�
prune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_prun_assert_greater_equal_y	S
Oprune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_identity_1
��Kprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert�
Rprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Rprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Rprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*Z
valueQBO BIx (prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp:0) = �
Rprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (prune_low_magnitude_fc2_prun/assert_greater_equal/y:0) = �
Kprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/AssertAssert�prune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_prun_assert_greater_equal_all[prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0[prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0[prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0�prune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_prun_assert_greater_equal_readvariableop[prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0prune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_prun_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Mprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_prun_assert_greater_equal_allL^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
Oprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityVprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity:output:0J^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Iprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/NoOpNoOpL^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "�
Oprune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_identity_1Xprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2�
Kprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/AssertKprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
,__inference_classifier_layer_call_fn_1596957

inputs
unknown:@
	unknown_0:
	unknown_1:	 
	unknown_2:
	unknown_3:
	unknown_4: 
	unknown_5:
	unknown_6:	 
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10:

unknown_11:

unknown_12:

unknown_13:	 

unknown_14:

unknown_15:

unknown_16: 

unknown_17:

unknown_18:(

unknown_19:(

unknown_20:(


unknown_21:

identity��StatefulPartitionedCall�
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
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*-
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_classifier_layer_call_and_return_conditional_losses_1596538o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������@: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�'
G__inference_classifier_layer_call_and_return_conditional_losses_1598649

inputs-
fc1_readvariableop_resource:@+
fc1_readvariableop_3_resource:>
4prune_low_magnitude_fc2_prun_readvariableop_resource:	 ;
)prune_low_magnitude_fc2_prun_cond_input_1:;
)prune_low_magnitude_fc2_prun_cond_input_2:3
)prune_low_magnitude_fc2_prun_cond_input_3: D
6prune_low_magnitude_fc2_prun_readvariableop_4_resource:>
4prune_low_magnitude_fc3_prun_readvariableop_resource:	 ;
)prune_low_magnitude_fc3_prun_cond_input_1:;
)prune_low_magnitude_fc3_prun_cond_input_2:3
)prune_low_magnitude_fc3_prun_cond_input_3: D
6prune_low_magnitude_fc3_prun_readvariableop_4_resource:?
-encoder_output_matmul_readvariableop_resource:<
.encoder_output_biasadd_readvariableop_resource:E
;prune_low_magnitude_fc4_prunedclass_readvariableop_resource:	 B
0prune_low_magnitude_fc4_prunedclass_cond_input_1:B
0prune_low_magnitude_fc4_prunedclass_cond_input_2::
0prune_low_magnitude_fc4_prunedclass_cond_input_3: K
=prune_low_magnitude_fc4_prunedclass_readvariableop_4_resource:3
!fc5_class_readvariableop_resource:(1
#fc5_class_readvariableop_3_resource:(8
&classifier_out_readvariableop_resource:(
6
(classifier_out_readvariableop_3_resource:

identity��classifier_out/ReadVariableOp�classifier_out/ReadVariableOp_1�classifier_out/ReadVariableOp_2�classifier_out/ReadVariableOp_3�classifier_out/ReadVariableOp_4�classifier_out/ReadVariableOp_5�4classifier_out/kernel/Regularizer/Abs/ReadVariableOp�%encoder_output/BiasAdd/ReadVariableOp�$encoder_output/MatMul/ReadVariableOp�fc1/ReadVariableOp�fc1/ReadVariableOp_1�fc1/ReadVariableOp_2�fc1/ReadVariableOp_3�fc1/ReadVariableOp_4�fc1/ReadVariableOp_5�fc5_class/ReadVariableOp�fc5_class/ReadVariableOp_1�fc5_class/ReadVariableOp_2�fc5_class/ReadVariableOp_3�fc5_class/ReadVariableOp_4�fc5_class/ReadVariableOp_5�/fc5_class/kernel/Regularizer/Abs/ReadVariableOp�-prune_low_magnitude_fc2_prun/AssignVariableOp�/prune_low_magnitude_fc2_prun/AssignVariableOp_1�8prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOp�5prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOp�/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp�1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1�+prune_low_magnitude_fc2_prun/ReadVariableOp�-prune_low_magnitude_fc2_prun/ReadVariableOp_1�-prune_low_magnitude_fc2_prun/ReadVariableOp_2�-prune_low_magnitude_fc2_prun/ReadVariableOp_3�-prune_low_magnitude_fc2_prun/ReadVariableOp_4�-prune_low_magnitude_fc2_prun/ReadVariableOp_5�-prune_low_magnitude_fc2_prun/ReadVariableOp_6�/prune_low_magnitude_fc2_prun/Sub/ReadVariableOp�Dprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard�@prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp�!prune_low_magnitude_fc2_prun/cond�-prune_low_magnitude_fc3_prun/AssignVariableOp�/prune_low_magnitude_fc3_prun/AssignVariableOp_1�8prune_low_magnitude_fc3_prun/GreaterEqual/ReadVariableOp�5prune_low_magnitude_fc3_prun/LessEqual/ReadVariableOp�/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp�1prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1�+prune_low_magnitude_fc3_prun/ReadVariableOp�-prune_low_magnitude_fc3_prun/ReadVariableOp_1�-prune_low_magnitude_fc3_prun/ReadVariableOp_2�-prune_low_magnitude_fc3_prun/ReadVariableOp_3�-prune_low_magnitude_fc3_prun/ReadVariableOp_4�-prune_low_magnitude_fc3_prun/ReadVariableOp_5�-prune_low_magnitude_fc3_prun/ReadVariableOp_6�/prune_low_magnitude_fc3_prun/Sub/ReadVariableOp�Dprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard�@prune_low_magnitude_fc3_prun/assert_greater_equal/ReadVariableOp�!prune_low_magnitude_fc3_prun/cond�4prune_low_magnitude_fc4_prunedclass/AssignVariableOp�6prune_low_magnitude_fc4_prunedclass/AssignVariableOp_1�?prune_low_magnitude_fc4_prunedclass/GreaterEqual/ReadVariableOp�<prune_low_magnitude_fc4_prunedclass/LessEqual/ReadVariableOp�6prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp�8prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1�2prune_low_magnitude_fc4_prunedclass/ReadVariableOp�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_6�6prune_low_magnitude_fc4_prunedclass/Sub/ReadVariableOp�Kprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard�Gprune_low_magnitude_fc4_prunedclass/assert_greater_equal/ReadVariableOp�(prune_low_magnitude_fc4_prunedclass/cond�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpK
	fc1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :K
	fc1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : W
fc1/PowPowfc1/Pow/x:output:0fc1/Pow/y:output:0*
T0*
_output_shapes
: M
fc1/CastCastfc1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: n
fc1/ReadVariableOpReadVariableOpfc1_readvariableop_resource*
_output_shapes

:@*
dtype0N
	fc1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ag
fc1/mulMulfc1/ReadVariableOp:value:0fc1/mul/y:output:0*
T0*
_output_shapes

:@Z
fc1/truedivRealDivfc1/mul:z:0fc1/Cast:y:0*
T0*
_output_shapes

:@H
fc1/NegNegfc1/truediv:z:0*
T0*
_output_shapes

:@L
	fc1/RoundRoundfc1/truediv:z:0*
T0*
_output_shapes

:@U
fc1/addAddV2fc1/Neg:y:0fc1/Round:y:0*
T0*
_output_shapes

:@V
fc1/StopGradientStopGradientfc1/add:z:0*
T0*
_output_shapes

:@g
	fc1/add_1AddV2fc1/truediv:z:0fc1/StopGradient:output:0*
T0*
_output_shapes

:@`
fc1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
fc1/clip_by_value/MinimumMinimumfc1/add_1:z:0$fc1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@X
fc1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
fc1/clip_by_valueMaximumfc1/clip_by_value/Minimum:z:0fc1/clip_by_value/y:output:0*
T0*
_output_shapes

:@^
	fc1/mul_1Mulfc1/Cast:y:0fc1/clip_by_value:z:0*
T0*
_output_shapes

:@T
fc1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aj
fc1/truediv_1RealDivfc1/mul_1:z:0fc1/truediv_1/y:output:0*
T0*
_output_shapes

:@P
fc1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	fc1/mul_2Mulfc1/mul_2/x:output:0fc1/truediv_1:z:0*
T0*
_output_shapes

:@p
fc1/ReadVariableOp_1ReadVariableOpfc1_readvariableop_resource*
_output_shapes

:@*
dtype0W
	fc1/Neg_1Negfc1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:@Y
	fc1/add_2AddV2fc1/Neg_1:y:0fc1/mul_2:z:0*
T0*
_output_shapes

:@P
fc1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
	fc1/mul_3Mulfc1/mul_3/x:output:0fc1/add_2:z:0*
T0*
_output_shapes

:@Z
fc1/StopGradient_1StopGradientfc1/mul_3:z:0*
T0*
_output_shapes

:@p
fc1/ReadVariableOp_2ReadVariableOpfc1_readvariableop_resource*
_output_shapes

:@*
dtype0v
	fc1/add_3AddV2fc1/ReadVariableOp_2:value:0fc1/StopGradient_1:output:0*
T0*
_output_shapes

:@]

fc1/MatMulMatMulinputsfc1/add_3:z:0*
T0*'
_output_shapes
:���������M
fc1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :M
fc1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
	fc1/Pow_1Powfc1/Pow_1/x:output:0fc1/Pow_1/y:output:0*
T0*
_output_shapes
: Q

fc1/Cast_1Castfc1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: n
fc1/ReadVariableOp_3ReadVariableOpfc1_readvariableop_3_resource*
_output_shapes
:*
dtype0P
fc1/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ai
	fc1/mul_4Mulfc1/ReadVariableOp_3:value:0fc1/mul_4/y:output:0*
T0*
_output_shapes
:\
fc1/truediv_2RealDivfc1/mul_4:z:0fc1/Cast_1:y:0*
T0*
_output_shapes
:H
	fc1/Neg_2Negfc1/truediv_2:z:0*
T0*
_output_shapes
:L
fc1/Round_1Roundfc1/truediv_2:z:0*
T0*
_output_shapes
:W
	fc1/add_4AddV2fc1/Neg_2:y:0fc1/Round_1:y:0*
T0*
_output_shapes
:V
fc1/StopGradient_2StopGradientfc1/add_4:z:0*
T0*
_output_shapes
:g
	fc1/add_5AddV2fc1/truediv_2:z:0fc1/StopGradient_2:output:0*
T0*
_output_shapes
:b
fc1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
fc1/clip_by_value_1/MinimumMinimumfc1/add_5:z:0&fc1/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:Z
fc1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
fc1/clip_by_value_1Maximumfc1/clip_by_value_1/Minimum:z:0fc1/clip_by_value_1/y:output:0*
T0*
_output_shapes
:^
	fc1/mul_5Mulfc1/Cast_1:y:0fc1/clip_by_value_1:z:0*
T0*
_output_shapes
:T
fc1/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Af
fc1/truediv_3RealDivfc1/mul_5:z:0fc1/truediv_3/y:output:0*
T0*
_output_shapes
:P
fc1/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
	fc1/mul_6Mulfc1/mul_6/x:output:0fc1/truediv_3:z:0*
T0*
_output_shapes
:n
fc1/ReadVariableOp_4ReadVariableOpfc1_readvariableop_3_resource*
_output_shapes
:*
dtype0S
	fc1/Neg_3Negfc1/ReadVariableOp_4:value:0*
T0*
_output_shapes
:U
	fc1/add_6AddV2fc1/Neg_3:y:0fc1/mul_6:z:0*
T0*
_output_shapes
:P
fc1/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
	fc1/mul_7Mulfc1/mul_7/x:output:0fc1/add_6:z:0*
T0*
_output_shapes
:V
fc1/StopGradient_3StopGradientfc1/mul_7:z:0*
T0*
_output_shapes
:n
fc1/ReadVariableOp_5ReadVariableOpfc1_readvariableop_3_resource*
_output_shapes
:*
dtype0r
	fc1/add_7AddV2fc1/ReadVariableOp_5:value:0fc1/StopGradient_3:output:0*
T0*
_output_shapes
:m
fc1/BiasAddBiasAddfc1/MatMul:product:0fc1/add_7:z:0*
T0*'
_output_shapes
:���������M
relu1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :M
relu1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :]
	relu1/PowPowrelu1/Pow/x:output:0relu1/Pow/y:output:0*
T0*
_output_shapes
: Q

relu1/CastCastrelu1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: O
relu1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :O
relu1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : c
relu1/Pow_1Powrelu1/Pow_1/x:output:0relu1/Pow_1/y:output:0*
T0*
_output_shapes
: U
relu1/Cast_1Castrelu1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: P
relu1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @P
relu1/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : ]
relu1/Cast_2Castrelu1/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: P
relu1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Y
	relu1/subSubrelu1/Cast_2:y:0relu1/sub/y:output:0*
T0*
_output_shapes
: X
relu1/Pow_2Powrelu1/Const:output:0relu1/sub:z:0*
T0*
_output_shapes
: V
relu1/sub_1Subrelu1/Cast_1:y:0relu1/Pow_2:z:0*
T0*
_output_shapes
: u
relu1/LessEqual	LessEqualfc1/BiasAdd:output:0relu1/sub_1:z:0*
T0*'
_output_shapes
:���������Z

relu1/ReluRelufc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������Y
relu1/ones_like/ShapeShapefc1/BiasAdd:output:0*
T0*
_output_shapes
:Z
relu1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
relu1/ones_likeFillrelu1/ones_like/Shape:output:0relu1/ones_like/Const:output:0*
T0*'
_output_shapes
:���������V
relu1/sub_2Subrelu1/Cast_1:y:0relu1/Pow_2:z:0*
T0*
_output_shapes
: m
	relu1/mulMulrelu1/ones_like:output:0relu1/sub_2:z:0*
T0*'
_output_shapes
:����������
relu1/SelectV2SelectV2relu1/LessEqual:z:0relu1/Relu:activations:0relu1/mul:z:0*
T0*'
_output_shapes
:���������j
relu1/mul_1Mulfc1/BiasAdd:output:0relu1/Cast:y:0*
T0*'
_output_shapes
:���������m
relu1/truedivRealDivrelu1/mul_1:z:0relu1/Cast_1:y:0*
T0*'
_output_shapes
:���������U
	relu1/NegNegrelu1/truediv:z:0*
T0*'
_output_shapes
:���������Y
relu1/RoundRoundrelu1/truediv:z:0*
T0*'
_output_shapes
:���������d
	relu1/addAddV2relu1/Neg:y:0relu1/Round:y:0*
T0*'
_output_shapes
:���������c
relu1/StopGradientStopGradientrelu1/add:z:0*
T0*'
_output_shapes
:���������v
relu1/add_1AddV2relu1/truediv:z:0relu1/StopGradient:output:0*
T0*'
_output_shapes
:���������m
relu1/truediv_1RealDivrelu1/add_1:z:0relu1/Cast:y:0*
T0*'
_output_shapes
:���������V
relu1/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
relu1/truediv_2RealDivrelu1/truediv_2/x:output:0relu1/Cast:y:0*
T0*
_output_shapes
: R
relu1/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?`
relu1/sub_3Subrelu1/sub_3/x:output:0relu1/truediv_2:z:0*
T0*
_output_shapes
: ~
relu1/clip_by_value/MinimumMinimumrelu1/truediv_1:z:0relu1/sub_3:z:0*
T0*'
_output_shapes
:���������Z
relu1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
relu1/clip_by_valueMaximumrelu1/clip_by_value/Minimum:z:0relu1/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������o
relu1/mul_2Mulrelu1/Cast_1:y:0relu1/clip_by_value:z:0*
T0*'
_output_shapes
:���������]
relu1/Neg_1Negrelu1/SelectV2:output:0*
T0*'
_output_shapes
:���������h
relu1/add_2AddV2relu1/Neg_1:y:0relu1/mul_2:z:0*
T0*'
_output_shapes
:���������R
relu1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?m
relu1/mul_3Mulrelu1/mul_3/x:output:0relu1/add_2:z:0*
T0*'
_output_shapes
:���������g
relu1/StopGradient_1StopGradientrelu1/mul_3:z:0*
T0*'
_output_shapes
:���������~
relu1/add_3AddV2relu1/SelectV2:output:0relu1/StopGradient_1:output:0*
T0*'
_output_shapes
:����������
+prune_low_magnitude_fc2_prun/ReadVariableOpReadVariableOp4prune_low_magnitude_fc2_prun_readvariableop_resource*
_output_shapes
: *
dtype0	d
"prune_low_magnitude_fc2_prun/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
 prune_low_magnitude_fc2_prun/addAddV23prune_low_magnitude_fc2_prun/ReadVariableOp:value:0+prune_low_magnitude_fc2_prun/add/y:output:0*
T0	*
_output_shapes
: �
-prune_low_magnitude_fc2_prun/AssignVariableOpAssignVariableOp4prune_low_magnitude_fc2_prun_readvariableop_resource$prune_low_magnitude_fc2_prun/add:z:0,^prune_low_magnitude_fc2_prun/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(q
#prune_low_magnitude_fc2_prun/updateNoOp.^prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes
 �
@prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOpReadVariableOp4prune_low_magnitude_fc2_prun_readvariableop_resource.^prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes
: *
dtype0	u
3prune_low_magnitude_fc2_prun/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
>prune_low_magnitude_fc2_prun/assert_greater_equal/GreaterEqualGreaterEqualHprune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp:value:0<prune_low_magnitude_fc2_prun/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: x
6prune_low_magnitude_fc2_prun/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
=prune_low_magnitude_fc2_prun/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
=prune_low_magnitude_fc2_prun/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
7prune_low_magnitude_fc2_prun/assert_greater_equal/rangeRangeFprune_low_magnitude_fc2_prun/assert_greater_equal/range/start:output:0?prune_low_magnitude_fc2_prun/assert_greater_equal/Rank:output:0Fprune_low_magnitude_fc2_prun/assert_greater_equal/range/delta:output:0*
_output_shapes
: �
5prune_low_magnitude_fc2_prun/assert_greater_equal/AllAllBprune_low_magnitude_fc2_prun/assert_greater_equal/GreaterEqual:z:0@prune_low_magnitude_fc2_prun/assert_greater_equal/range:output:0*
_output_shapes
: �
>prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
@prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
@prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*Z
valueQBO BIx (prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp:0) = �
@prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (prune_low_magnitude_fc2_prun/assert_greater_equal/y:0) = �
Dprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuardIf>prune_low_magnitude_fc2_prun/assert_greater_equal/All:output:0>prune_low_magnitude_fc2_prun/assert_greater_equal/All:output:0Hprune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp:value:0<prune_low_magnitude_fc2_prun/assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *e
else_branchVRT
Rprune_low_magnitude_fc2_prun_assert_greater_equal_Assert_AssertGuard_false_1597737*
output_shapes
: *d
then_branchURS
Qprune_low_magnitude_fc2_prun_assert_greater_equal_Assert_AssertGuard_true_1597736�
Mprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/IdentityIdentityMprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
8prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOpReadVariableOp4prune_low_magnitude_fc2_prun_readvariableop_resource.^prune_low_magnitude_fc2_prun/AssignVariableOpN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
+prune_low_magnitude_fc2_prun/GreaterEqual/yConstN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
)prune_low_magnitude_fc2_prun/GreaterEqualGreaterEqual@prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOp:value:04prune_low_magnitude_fc2_prun/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
5prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOpReadVariableOp4prune_low_magnitude_fc2_prun_readvariableop_resource.^prune_low_magnitude_fc2_prun/AssignVariableOpN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
(prune_low_magnitude_fc2_prun/LessEqual/yConstN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�N�
&prune_low_magnitude_fc2_prun/LessEqual	LessEqual=prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOp:value:01prune_low_magnitude_fc2_prun/LessEqual/y:output:0*
T0	*
_output_shapes
: �
#prune_low_magnitude_fc2_prun/Less/xConstN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N�
#prune_low_magnitude_fc2_prun/Less/yConstN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : �
!prune_low_magnitude_fc2_prun/LessLess,prune_low_magnitude_fc2_prun/Less/x:output:0,prune_low_magnitude_fc2_prun/Less/y:output:0*
T0*
_output_shapes
: �
&prune_low_magnitude_fc2_prun/LogicalOr	LogicalOr*prune_low_magnitude_fc2_prun/LessEqual:z:0%prune_low_magnitude_fc2_prun/Less:z:0*
_output_shapes
: �
'prune_low_magnitude_fc2_prun/LogicalAnd
LogicalAnd-prune_low_magnitude_fc2_prun/GreaterEqual:z:0*prune_low_magnitude_fc2_prun/LogicalOr:z:0*
_output_shapes
: �
/prune_low_magnitude_fc2_prun/Sub/ReadVariableOpReadVariableOp4prune_low_magnitude_fc2_prun_readvariableop_resource.^prune_low_magnitude_fc2_prun/AssignVariableOpN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
"prune_low_magnitude_fc2_prun/Sub/yConstN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
 prune_low_magnitude_fc2_prun/SubSub7prune_low_magnitude_fc2_prun/Sub/ReadVariableOp:value:0+prune_low_magnitude_fc2_prun/Sub/y:output:0*
T0	*
_output_shapes
: �
'prune_low_magnitude_fc2_prun/FloorMod/yConstN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd�
%prune_low_magnitude_fc2_prun/FloorModFloorMod$prune_low_magnitude_fc2_prun/Sub:z:00prune_low_magnitude_fc2_prun/FloorMod/y:output:0*
T0	*
_output_shapes
: �
$prune_low_magnitude_fc2_prun/Equal/yConstN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R �
"prune_low_magnitude_fc2_prun/EqualEqual)prune_low_magnitude_fc2_prun/FloorMod:z:0-prune_low_magnitude_fc2_prun/Equal/y:output:0*
T0	*
_output_shapes
: �
)prune_low_magnitude_fc2_prun/LogicalAnd_1
LogicalAnd+prune_low_magnitude_fc2_prun/LogicalAnd:z:0&prune_low_magnitude_fc2_prun/Equal:z:0*
_output_shapes
: �
"prune_low_magnitude_fc2_prun/ConstConstN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��L?�
!prune_low_magnitude_fc2_prun/condIf-prune_low_magnitude_fc2_prun/LogicalAnd_1:z:04prune_low_magnitude_fc2_prun_readvariableop_resource)prune_low_magnitude_fc2_prun_cond_input_1)prune_low_magnitude_fc2_prun_cond_input_2)prune_low_magnitude_fc2_prun_cond_input_3-prune_low_magnitude_fc2_prun/LogicalAnd_1:z:0.^prune_low_magnitude_fc2_prun/AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*B
else_branch3R1
/prune_low_magnitude_fc2_prun_cond_false_1597777*
output_shapes
: *A
then_branch2R0
.prune_low_magnitude_fc2_prun_cond_true_1597776�
*prune_low_magnitude_fc2_prun/cond/IdentityIdentity*prune_low_magnitude_fc2_prun/cond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
%prune_low_magnitude_fc2_prun/update_1NoOpN^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity+^prune_low_magnitude_fc2_prun/cond/Identity*
_output_shapes
 �
/prune_low_magnitude_fc2_prun/Mul/ReadVariableOpReadVariableOp)prune_low_magnitude_fc2_prun_cond_input_1*
_output_shapes

:*
dtype0�
1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1ReadVariableOp)prune_low_magnitude_fc2_prun_cond_input_2"^prune_low_magnitude_fc2_prun/cond*
_output_shapes

:*
dtype0�
 prune_low_magnitude_fc2_prun/MulMul7prune_low_magnitude_fc2_prun/Mul/ReadVariableOp:value:09prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
/prune_low_magnitude_fc2_prun/AssignVariableOp_1AssignVariableOp)prune_low_magnitude_fc2_prun_cond_input_1$prune_low_magnitude_fc2_prun/Mul:z:00^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp"^prune_low_magnitude_fc2_prun/cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
'prune_low_magnitude_fc2_prun/group_depsNoOp0^prune_low_magnitude_fc2_prun/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
)prune_low_magnitude_fc2_prun/group_deps_1NoOp(^prune_low_magnitude_fc2_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 d
"prune_low_magnitude_fc2_prun/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :d
"prune_low_magnitude_fc2_prun/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
 prune_low_magnitude_fc2_prun/PowPow+prune_low_magnitude_fc2_prun/Pow/x:output:0+prune_low_magnitude_fc2_prun/Pow/y:output:0*
T0*
_output_shapes
: 
!prune_low_magnitude_fc2_prun/CastCast$prune_low_magnitude_fc2_prun/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
-prune_low_magnitude_fc2_prun/ReadVariableOp_1ReadVariableOp)prune_low_magnitude_fc2_prun_cond_input_10^prune_low_magnitude_fc2_prun/AssignVariableOp_1*
_output_shapes

:*
dtype0i
$prune_low_magnitude_fc2_prun/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
"prune_low_magnitude_fc2_prun/mul_1Mul5prune_low_magnitude_fc2_prun/ReadVariableOp_1:value:0-prune_low_magnitude_fc2_prun/mul_1/y:output:0*
T0*
_output_shapes

:�
$prune_low_magnitude_fc2_prun/truedivRealDiv&prune_low_magnitude_fc2_prun/mul_1:z:0%prune_low_magnitude_fc2_prun/Cast:y:0*
T0*
_output_shapes

:z
 prune_low_magnitude_fc2_prun/NegNeg(prune_low_magnitude_fc2_prun/truediv:z:0*
T0*
_output_shapes

:~
"prune_low_magnitude_fc2_prun/RoundRound(prune_low_magnitude_fc2_prun/truediv:z:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc2_prun/add_1AddV2$prune_low_magnitude_fc2_prun/Neg:y:0&prune_low_magnitude_fc2_prun/Round:y:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc2_prun/StopGradientStopGradient&prune_low_magnitude_fc2_prun/add_1:z:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc2_prun/add_2AddV2(prune_low_magnitude_fc2_prun/truediv:z:02prune_low_magnitude_fc2_prun/StopGradient:output:0*
T0*
_output_shapes

:y
4prune_low_magnitude_fc2_prun/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
2prune_low_magnitude_fc2_prun/clip_by_value/MinimumMinimum&prune_low_magnitude_fc2_prun/add_2:z:0=prune_low_magnitude_fc2_prun/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:q
,prune_low_magnitude_fc2_prun/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
*prune_low_magnitude_fc2_prun/clip_by_valueMaximum6prune_low_magnitude_fc2_prun/clip_by_value/Minimum:z:05prune_low_magnitude_fc2_prun/clip_by_value/y:output:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc2_prun/mul_2Mul%prune_low_magnitude_fc2_prun/Cast:y:0.prune_low_magnitude_fc2_prun/clip_by_value:z:0*
T0*
_output_shapes

:m
(prune_low_magnitude_fc2_prun/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
&prune_low_magnitude_fc2_prun/truediv_1RealDiv&prune_low_magnitude_fc2_prun/mul_2:z:01prune_low_magnitude_fc2_prun/truediv_1/y:output:0*
T0*
_output_shapes

:i
$prune_low_magnitude_fc2_prun/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc2_prun/mul_3Mul-prune_low_magnitude_fc2_prun/mul_3/x:output:0*prune_low_magnitude_fc2_prun/truediv_1:z:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc2_prun/ReadVariableOp_2ReadVariableOp)prune_low_magnitude_fc2_prun_cond_input_10^prune_low_magnitude_fc2_prun/AssignVariableOp_1*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc2_prun/Neg_1Neg5prune_low_magnitude_fc2_prun/ReadVariableOp_2:value:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc2_prun/add_3AddV2&prune_low_magnitude_fc2_prun/Neg_1:y:0&prune_low_magnitude_fc2_prun/mul_3:z:0*
T0*
_output_shapes

:i
$prune_low_magnitude_fc2_prun/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc2_prun/mul_4Mul-prune_low_magnitude_fc2_prun/mul_4/x:output:0&prune_low_magnitude_fc2_prun/add_3:z:0*
T0*
_output_shapes

:�
+prune_low_magnitude_fc2_prun/StopGradient_1StopGradient&prune_low_magnitude_fc2_prun/mul_4:z:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc2_prun/ReadVariableOp_3ReadVariableOp)prune_low_magnitude_fc2_prun_cond_input_10^prune_low_magnitude_fc2_prun/AssignVariableOp_1*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc2_prun/add_4AddV25prune_low_magnitude_fc2_prun/ReadVariableOp_3:value:04prune_low_magnitude_fc2_prun/StopGradient_1:output:0*
T0*
_output_shapes

:�
#prune_low_magnitude_fc2_prun/MatMulMatMulrelu1/add_3:z:0&prune_low_magnitude_fc2_prun/add_4:z:0*
T0*'
_output_shapes
:���������f
$prune_low_magnitude_fc2_prun/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :f
$prune_low_magnitude_fc2_prun/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
"prune_low_magnitude_fc2_prun/Pow_1Pow-prune_low_magnitude_fc2_prun/Pow_1/x:output:0-prune_low_magnitude_fc2_prun/Pow_1/y:output:0*
T0*
_output_shapes
: �
#prune_low_magnitude_fc2_prun/Cast_1Cast&prune_low_magnitude_fc2_prun/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
-prune_low_magnitude_fc2_prun/ReadVariableOp_4ReadVariableOp6prune_low_magnitude_fc2_prun_readvariableop_4_resource*
_output_shapes
:*
dtype0i
$prune_low_magnitude_fc2_prun/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
"prune_low_magnitude_fc2_prun/mul_5Mul5prune_low_magnitude_fc2_prun/ReadVariableOp_4:value:0-prune_low_magnitude_fc2_prun/mul_5/y:output:0*
T0*
_output_shapes
:�
&prune_low_magnitude_fc2_prun/truediv_2RealDiv&prune_low_magnitude_fc2_prun/mul_5:z:0'prune_low_magnitude_fc2_prun/Cast_1:y:0*
T0*
_output_shapes
:z
"prune_low_magnitude_fc2_prun/Neg_2Neg*prune_low_magnitude_fc2_prun/truediv_2:z:0*
T0*
_output_shapes
:~
$prune_low_magnitude_fc2_prun/Round_1Round*prune_low_magnitude_fc2_prun/truediv_2:z:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc2_prun/add_5AddV2&prune_low_magnitude_fc2_prun/Neg_2:y:0(prune_low_magnitude_fc2_prun/Round_1:y:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc2_prun/StopGradient_2StopGradient&prune_low_magnitude_fc2_prun/add_5:z:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc2_prun/add_6AddV2*prune_low_magnitude_fc2_prun/truediv_2:z:04prune_low_magnitude_fc2_prun/StopGradient_2:output:0*
T0*
_output_shapes
:{
6prune_low_magnitude_fc2_prun/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
4prune_low_magnitude_fc2_prun/clip_by_value_1/MinimumMinimum&prune_low_magnitude_fc2_prun/add_6:z:0?prune_low_magnitude_fc2_prun/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:s
.prune_low_magnitude_fc2_prun/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
,prune_low_magnitude_fc2_prun/clip_by_value_1Maximum8prune_low_magnitude_fc2_prun/clip_by_value_1/Minimum:z:07prune_low_magnitude_fc2_prun/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc2_prun/mul_6Mul'prune_low_magnitude_fc2_prun/Cast_1:y:00prune_low_magnitude_fc2_prun/clip_by_value_1:z:0*
T0*
_output_shapes
:m
(prune_low_magnitude_fc2_prun/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
&prune_low_magnitude_fc2_prun/truediv_3RealDiv&prune_low_magnitude_fc2_prun/mul_6:z:01prune_low_magnitude_fc2_prun/truediv_3/y:output:0*
T0*
_output_shapes
:i
$prune_low_magnitude_fc2_prun/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc2_prun/mul_7Mul-prune_low_magnitude_fc2_prun/mul_7/x:output:0*prune_low_magnitude_fc2_prun/truediv_3:z:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc2_prun/ReadVariableOp_5ReadVariableOp6prune_low_magnitude_fc2_prun_readvariableop_4_resource*
_output_shapes
:*
dtype0�
"prune_low_magnitude_fc2_prun/Neg_3Neg5prune_low_magnitude_fc2_prun/ReadVariableOp_5:value:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc2_prun/add_7AddV2&prune_low_magnitude_fc2_prun/Neg_3:y:0&prune_low_magnitude_fc2_prun/mul_7:z:0*
T0*
_output_shapes
:i
$prune_low_magnitude_fc2_prun/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc2_prun/mul_8Mul-prune_low_magnitude_fc2_prun/mul_8/x:output:0&prune_low_magnitude_fc2_prun/add_7:z:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc2_prun/StopGradient_3StopGradient&prune_low_magnitude_fc2_prun/mul_8:z:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc2_prun/ReadVariableOp_6ReadVariableOp6prune_low_magnitude_fc2_prun_readvariableop_4_resource*
_output_shapes
:*
dtype0�
"prune_low_magnitude_fc2_prun/add_8AddV25prune_low_magnitude_fc2_prun/ReadVariableOp_6:value:04prune_low_magnitude_fc2_prun/StopGradient_3:output:0*
T0*
_output_shapes
:�
$prune_low_magnitude_fc2_prun/BiasAddBiasAdd-prune_low_magnitude_fc2_prun/MatMul:product:0&prune_low_magnitude_fc2_prun/add_8:z:0*
T0*'
_output_shapes
:���������M
relu2/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :M
relu2/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :]
	relu2/PowPowrelu2/Pow/x:output:0relu2/Pow/y:output:0*
T0*
_output_shapes
: Q

relu2/CastCastrelu2/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: O
relu2/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :O
relu2/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : c
relu2/Pow_1Powrelu2/Pow_1/x:output:0relu2/Pow_1/y:output:0*
T0*
_output_shapes
: U
relu2/Cast_1Castrelu2/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: P
relu2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @P
relu2/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : ]
relu2/Cast_2Castrelu2/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: P
relu2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Y
	relu2/subSubrelu2/Cast_2:y:0relu2/sub/y:output:0*
T0*
_output_shapes
: X
relu2/Pow_2Powrelu2/Const:output:0relu2/sub:z:0*
T0*
_output_shapes
: V
relu2/sub_1Subrelu2/Cast_1:y:0relu2/Pow_2:z:0*
T0*
_output_shapes
: �
relu2/LessEqual	LessEqual-prune_low_magnitude_fc2_prun/BiasAdd:output:0relu2/sub_1:z:0*
T0*'
_output_shapes
:���������s

relu2/ReluRelu-prune_low_magnitude_fc2_prun/BiasAdd:output:0*
T0*'
_output_shapes
:���������r
relu2/ones_like/ShapeShape-prune_low_magnitude_fc2_prun/BiasAdd:output:0*
T0*
_output_shapes
:Z
relu2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
relu2/ones_likeFillrelu2/ones_like/Shape:output:0relu2/ones_like/Const:output:0*
T0*'
_output_shapes
:���������V
relu2/sub_2Subrelu2/Cast_1:y:0relu2/Pow_2:z:0*
T0*
_output_shapes
: m
	relu2/mulMulrelu2/ones_like:output:0relu2/sub_2:z:0*
T0*'
_output_shapes
:����������
relu2/SelectV2SelectV2relu2/LessEqual:z:0relu2/Relu:activations:0relu2/mul:z:0*
T0*'
_output_shapes
:����������
relu2/mul_1Mul-prune_low_magnitude_fc2_prun/BiasAdd:output:0relu2/Cast:y:0*
T0*'
_output_shapes
:���������m
relu2/truedivRealDivrelu2/mul_1:z:0relu2/Cast_1:y:0*
T0*'
_output_shapes
:���������U
	relu2/NegNegrelu2/truediv:z:0*
T0*'
_output_shapes
:���������Y
relu2/RoundRoundrelu2/truediv:z:0*
T0*'
_output_shapes
:���������d
	relu2/addAddV2relu2/Neg:y:0relu2/Round:y:0*
T0*'
_output_shapes
:���������c
relu2/StopGradientStopGradientrelu2/add:z:0*
T0*'
_output_shapes
:���������v
relu2/add_1AddV2relu2/truediv:z:0relu2/StopGradient:output:0*
T0*'
_output_shapes
:���������m
relu2/truediv_1RealDivrelu2/add_1:z:0relu2/Cast:y:0*
T0*'
_output_shapes
:���������V
relu2/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
relu2/truediv_2RealDivrelu2/truediv_2/x:output:0relu2/Cast:y:0*
T0*
_output_shapes
: R
relu2/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?`
relu2/sub_3Subrelu2/sub_3/x:output:0relu2/truediv_2:z:0*
T0*
_output_shapes
: ~
relu2/clip_by_value/MinimumMinimumrelu2/truediv_1:z:0relu2/sub_3:z:0*
T0*'
_output_shapes
:���������Z
relu2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
relu2/clip_by_valueMaximumrelu2/clip_by_value/Minimum:z:0relu2/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������o
relu2/mul_2Mulrelu2/Cast_1:y:0relu2/clip_by_value:z:0*
T0*'
_output_shapes
:���������]
relu2/Neg_1Negrelu2/SelectV2:output:0*
T0*'
_output_shapes
:���������h
relu2/add_2AddV2relu2/Neg_1:y:0relu2/mul_2:z:0*
T0*'
_output_shapes
:���������R
relu2/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?m
relu2/mul_3Mulrelu2/mul_3/x:output:0relu2/add_2:z:0*
T0*'
_output_shapes
:���������g
relu2/StopGradient_1StopGradientrelu2/mul_3:z:0*
T0*'
_output_shapes
:���������~
relu2/add_3AddV2relu2/SelectV2:output:0relu2/StopGradient_1:output:0*
T0*'
_output_shapes
:����������
+prune_low_magnitude_fc3_prun/ReadVariableOpReadVariableOp4prune_low_magnitude_fc3_prun_readvariableop_resource*
_output_shapes
: *
dtype0	d
"prune_low_magnitude_fc3_prun/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
 prune_low_magnitude_fc3_prun/addAddV23prune_low_magnitude_fc3_prun/ReadVariableOp:value:0+prune_low_magnitude_fc3_prun/add/y:output:0*
T0	*
_output_shapes
: �
-prune_low_magnitude_fc3_prun/AssignVariableOpAssignVariableOp4prune_low_magnitude_fc3_prun_readvariableop_resource$prune_low_magnitude_fc3_prun/add:z:0,^prune_low_magnitude_fc3_prun/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(q
#prune_low_magnitude_fc3_prun/updateNoOp.^prune_low_magnitude_fc3_prun/AssignVariableOp*
_output_shapes
 �
@prune_low_magnitude_fc3_prun/assert_greater_equal/ReadVariableOpReadVariableOp4prune_low_magnitude_fc3_prun_readvariableop_resource.^prune_low_magnitude_fc3_prun/AssignVariableOp*
_output_shapes
: *
dtype0	u
3prune_low_magnitude_fc3_prun/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
>prune_low_magnitude_fc3_prun/assert_greater_equal/GreaterEqualGreaterEqualHprune_low_magnitude_fc3_prun/assert_greater_equal/ReadVariableOp:value:0<prune_low_magnitude_fc3_prun/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: x
6prune_low_magnitude_fc3_prun/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
=prune_low_magnitude_fc3_prun/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
=prune_low_magnitude_fc3_prun/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
7prune_low_magnitude_fc3_prun/assert_greater_equal/rangeRangeFprune_low_magnitude_fc3_prun/assert_greater_equal/range/start:output:0?prune_low_magnitude_fc3_prun/assert_greater_equal/Rank:output:0Fprune_low_magnitude_fc3_prun/assert_greater_equal/range/delta:output:0*
_output_shapes
: �
5prune_low_magnitude_fc3_prun/assert_greater_equal/AllAllBprune_low_magnitude_fc3_prun/assert_greater_equal/GreaterEqual:z:0@prune_low_magnitude_fc3_prun/assert_greater_equal/range:output:0*
_output_shapes
: �
>prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
@prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
@prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*Z
valueQBO BIx (prune_low_magnitude_fc3_prun/assert_greater_equal/ReadVariableOp:0) = �
@prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (prune_low_magnitude_fc3_prun/assert_greater_equal/y:0) = �
Dprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuardIf>prune_low_magnitude_fc3_prun/assert_greater_equal/All:output:0>prune_low_magnitude_fc3_prun/assert_greater_equal/All:output:0Hprune_low_magnitude_fc3_prun/assert_greater_equal/ReadVariableOp:value:0<prune_low_magnitude_fc3_prun/assert_greater_equal/y:output:0E^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *e
else_branchVRT
Rprune_low_magnitude_fc3_prun_assert_greater_equal_Assert_AssertGuard_false_1597980*
output_shapes
: *d
then_branchURS
Qprune_low_magnitude_fc3_prun_assert_greater_equal_Assert_AssertGuard_true_1597979�
Mprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/IdentityIdentityMprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
8prune_low_magnitude_fc3_prun/GreaterEqual/ReadVariableOpReadVariableOp4prune_low_magnitude_fc3_prun_readvariableop_resource.^prune_low_magnitude_fc3_prun/AssignVariableOpN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
+prune_low_magnitude_fc3_prun/GreaterEqual/yConstN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
)prune_low_magnitude_fc3_prun/GreaterEqualGreaterEqual@prune_low_magnitude_fc3_prun/GreaterEqual/ReadVariableOp:value:04prune_low_magnitude_fc3_prun/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
5prune_low_magnitude_fc3_prun/LessEqual/ReadVariableOpReadVariableOp4prune_low_magnitude_fc3_prun_readvariableop_resource.^prune_low_magnitude_fc3_prun/AssignVariableOpN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
(prune_low_magnitude_fc3_prun/LessEqual/yConstN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�N�
&prune_low_magnitude_fc3_prun/LessEqual	LessEqual=prune_low_magnitude_fc3_prun/LessEqual/ReadVariableOp:value:01prune_low_magnitude_fc3_prun/LessEqual/y:output:0*
T0	*
_output_shapes
: �
#prune_low_magnitude_fc3_prun/Less/xConstN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N�
#prune_low_magnitude_fc3_prun/Less/yConstN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : �
!prune_low_magnitude_fc3_prun/LessLess,prune_low_magnitude_fc3_prun/Less/x:output:0,prune_low_magnitude_fc3_prun/Less/y:output:0*
T0*
_output_shapes
: �
&prune_low_magnitude_fc3_prun/LogicalOr	LogicalOr*prune_low_magnitude_fc3_prun/LessEqual:z:0%prune_low_magnitude_fc3_prun/Less:z:0*
_output_shapes
: �
'prune_low_magnitude_fc3_prun/LogicalAnd
LogicalAnd-prune_low_magnitude_fc3_prun/GreaterEqual:z:0*prune_low_magnitude_fc3_prun/LogicalOr:z:0*
_output_shapes
: �
/prune_low_magnitude_fc3_prun/Sub/ReadVariableOpReadVariableOp4prune_low_magnitude_fc3_prun_readvariableop_resource.^prune_low_magnitude_fc3_prun/AssignVariableOpN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
"prune_low_magnitude_fc3_prun/Sub/yConstN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
 prune_low_magnitude_fc3_prun/SubSub7prune_low_magnitude_fc3_prun/Sub/ReadVariableOp:value:0+prune_low_magnitude_fc3_prun/Sub/y:output:0*
T0	*
_output_shapes
: �
'prune_low_magnitude_fc3_prun/FloorMod/yConstN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd�
%prune_low_magnitude_fc3_prun/FloorModFloorMod$prune_low_magnitude_fc3_prun/Sub:z:00prune_low_magnitude_fc3_prun/FloorMod/y:output:0*
T0	*
_output_shapes
: �
$prune_low_magnitude_fc3_prun/Equal/yConstN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R �
"prune_low_magnitude_fc3_prun/EqualEqual)prune_low_magnitude_fc3_prun/FloorMod:z:0-prune_low_magnitude_fc3_prun/Equal/y:output:0*
T0	*
_output_shapes
: �
)prune_low_magnitude_fc3_prun/LogicalAnd_1
LogicalAnd+prune_low_magnitude_fc3_prun/LogicalAnd:z:0&prune_low_magnitude_fc3_prun/Equal:z:0*
_output_shapes
: �
"prune_low_magnitude_fc3_prun/ConstConstN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��L?�
!prune_low_magnitude_fc3_prun/condIf-prune_low_magnitude_fc3_prun/LogicalAnd_1:z:04prune_low_magnitude_fc3_prun_readvariableop_resource)prune_low_magnitude_fc3_prun_cond_input_1)prune_low_magnitude_fc3_prun_cond_input_2)prune_low_magnitude_fc3_prun_cond_input_3-prune_low_magnitude_fc3_prun/LogicalAnd_1:z:0.^prune_low_magnitude_fc3_prun/AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*B
else_branch3R1
/prune_low_magnitude_fc3_prun_cond_false_1598020*
output_shapes
: *A
then_branch2R0
.prune_low_magnitude_fc3_prun_cond_true_1598019�
*prune_low_magnitude_fc3_prun/cond/IdentityIdentity*prune_low_magnitude_fc3_prun/cond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
%prune_low_magnitude_fc3_prun/update_1NoOpN^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity+^prune_low_magnitude_fc3_prun/cond/Identity*
_output_shapes
 �
/prune_low_magnitude_fc3_prun/Mul/ReadVariableOpReadVariableOp)prune_low_magnitude_fc3_prun_cond_input_1*
_output_shapes

:*
dtype0�
1prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1ReadVariableOp)prune_low_magnitude_fc3_prun_cond_input_2"^prune_low_magnitude_fc3_prun/cond*
_output_shapes

:*
dtype0�
 prune_low_magnitude_fc3_prun/MulMul7prune_low_magnitude_fc3_prun/Mul/ReadVariableOp:value:09prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
/prune_low_magnitude_fc3_prun/AssignVariableOp_1AssignVariableOp)prune_low_magnitude_fc3_prun_cond_input_1$prune_low_magnitude_fc3_prun/Mul:z:00^prune_low_magnitude_fc3_prun/Mul/ReadVariableOp"^prune_low_magnitude_fc3_prun/cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
'prune_low_magnitude_fc3_prun/group_depsNoOp0^prune_low_magnitude_fc3_prun/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
)prune_low_magnitude_fc3_prun/group_deps_1NoOp(^prune_low_magnitude_fc3_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 d
"prune_low_magnitude_fc3_prun/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :d
"prune_low_magnitude_fc3_prun/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
 prune_low_magnitude_fc3_prun/PowPow+prune_low_magnitude_fc3_prun/Pow/x:output:0+prune_low_magnitude_fc3_prun/Pow/y:output:0*
T0*
_output_shapes
: 
!prune_low_magnitude_fc3_prun/CastCast$prune_low_magnitude_fc3_prun/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
-prune_low_magnitude_fc3_prun/ReadVariableOp_1ReadVariableOp)prune_low_magnitude_fc3_prun_cond_input_10^prune_low_magnitude_fc3_prun/AssignVariableOp_1*
_output_shapes

:*
dtype0i
$prune_low_magnitude_fc3_prun/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
"prune_low_magnitude_fc3_prun/mul_1Mul5prune_low_magnitude_fc3_prun/ReadVariableOp_1:value:0-prune_low_magnitude_fc3_prun/mul_1/y:output:0*
T0*
_output_shapes

:�
$prune_low_magnitude_fc3_prun/truedivRealDiv&prune_low_magnitude_fc3_prun/mul_1:z:0%prune_low_magnitude_fc3_prun/Cast:y:0*
T0*
_output_shapes

:z
 prune_low_magnitude_fc3_prun/NegNeg(prune_low_magnitude_fc3_prun/truediv:z:0*
T0*
_output_shapes

:~
"prune_low_magnitude_fc3_prun/RoundRound(prune_low_magnitude_fc3_prun/truediv:z:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc3_prun/add_1AddV2$prune_low_magnitude_fc3_prun/Neg:y:0&prune_low_magnitude_fc3_prun/Round:y:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc3_prun/StopGradientStopGradient&prune_low_magnitude_fc3_prun/add_1:z:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc3_prun/add_2AddV2(prune_low_magnitude_fc3_prun/truediv:z:02prune_low_magnitude_fc3_prun/StopGradient:output:0*
T0*
_output_shapes

:y
4prune_low_magnitude_fc3_prun/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
2prune_low_magnitude_fc3_prun/clip_by_value/MinimumMinimum&prune_low_magnitude_fc3_prun/add_2:z:0=prune_low_magnitude_fc3_prun/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:q
,prune_low_magnitude_fc3_prun/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
*prune_low_magnitude_fc3_prun/clip_by_valueMaximum6prune_low_magnitude_fc3_prun/clip_by_value/Minimum:z:05prune_low_magnitude_fc3_prun/clip_by_value/y:output:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc3_prun/mul_2Mul%prune_low_magnitude_fc3_prun/Cast:y:0.prune_low_magnitude_fc3_prun/clip_by_value:z:0*
T0*
_output_shapes

:m
(prune_low_magnitude_fc3_prun/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
&prune_low_magnitude_fc3_prun/truediv_1RealDiv&prune_low_magnitude_fc3_prun/mul_2:z:01prune_low_magnitude_fc3_prun/truediv_1/y:output:0*
T0*
_output_shapes

:i
$prune_low_magnitude_fc3_prun/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc3_prun/mul_3Mul-prune_low_magnitude_fc3_prun/mul_3/x:output:0*prune_low_magnitude_fc3_prun/truediv_1:z:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc3_prun/ReadVariableOp_2ReadVariableOp)prune_low_magnitude_fc3_prun_cond_input_10^prune_low_magnitude_fc3_prun/AssignVariableOp_1*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc3_prun/Neg_1Neg5prune_low_magnitude_fc3_prun/ReadVariableOp_2:value:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc3_prun/add_3AddV2&prune_low_magnitude_fc3_prun/Neg_1:y:0&prune_low_magnitude_fc3_prun/mul_3:z:0*
T0*
_output_shapes

:i
$prune_low_magnitude_fc3_prun/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc3_prun/mul_4Mul-prune_low_magnitude_fc3_prun/mul_4/x:output:0&prune_low_magnitude_fc3_prun/add_3:z:0*
T0*
_output_shapes

:�
+prune_low_magnitude_fc3_prun/StopGradient_1StopGradient&prune_low_magnitude_fc3_prun/mul_4:z:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc3_prun/ReadVariableOp_3ReadVariableOp)prune_low_magnitude_fc3_prun_cond_input_10^prune_low_magnitude_fc3_prun/AssignVariableOp_1*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc3_prun/add_4AddV25prune_low_magnitude_fc3_prun/ReadVariableOp_3:value:04prune_low_magnitude_fc3_prun/StopGradient_1:output:0*
T0*
_output_shapes

:�
#prune_low_magnitude_fc3_prun/MatMulMatMulrelu2/add_3:z:0&prune_low_magnitude_fc3_prun/add_4:z:0*
T0*'
_output_shapes
:���������f
$prune_low_magnitude_fc3_prun/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :f
$prune_low_magnitude_fc3_prun/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
"prune_low_magnitude_fc3_prun/Pow_1Pow-prune_low_magnitude_fc3_prun/Pow_1/x:output:0-prune_low_magnitude_fc3_prun/Pow_1/y:output:0*
T0*
_output_shapes
: �
#prune_low_magnitude_fc3_prun/Cast_1Cast&prune_low_magnitude_fc3_prun/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
-prune_low_magnitude_fc3_prun/ReadVariableOp_4ReadVariableOp6prune_low_magnitude_fc3_prun_readvariableop_4_resource*
_output_shapes
:*
dtype0i
$prune_low_magnitude_fc3_prun/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
"prune_low_magnitude_fc3_prun/mul_5Mul5prune_low_magnitude_fc3_prun/ReadVariableOp_4:value:0-prune_low_magnitude_fc3_prun/mul_5/y:output:0*
T0*
_output_shapes
:�
&prune_low_magnitude_fc3_prun/truediv_2RealDiv&prune_low_magnitude_fc3_prun/mul_5:z:0'prune_low_magnitude_fc3_prun/Cast_1:y:0*
T0*
_output_shapes
:z
"prune_low_magnitude_fc3_prun/Neg_2Neg*prune_low_magnitude_fc3_prun/truediv_2:z:0*
T0*
_output_shapes
:~
$prune_low_magnitude_fc3_prun/Round_1Round*prune_low_magnitude_fc3_prun/truediv_2:z:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc3_prun/add_5AddV2&prune_low_magnitude_fc3_prun/Neg_2:y:0(prune_low_magnitude_fc3_prun/Round_1:y:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc3_prun/StopGradient_2StopGradient&prune_low_magnitude_fc3_prun/add_5:z:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc3_prun/add_6AddV2*prune_low_magnitude_fc3_prun/truediv_2:z:04prune_low_magnitude_fc3_prun/StopGradient_2:output:0*
T0*
_output_shapes
:{
6prune_low_magnitude_fc3_prun/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
4prune_low_magnitude_fc3_prun/clip_by_value_1/MinimumMinimum&prune_low_magnitude_fc3_prun/add_6:z:0?prune_low_magnitude_fc3_prun/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:s
.prune_low_magnitude_fc3_prun/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
,prune_low_magnitude_fc3_prun/clip_by_value_1Maximum8prune_low_magnitude_fc3_prun/clip_by_value_1/Minimum:z:07prune_low_magnitude_fc3_prun/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc3_prun/mul_6Mul'prune_low_magnitude_fc3_prun/Cast_1:y:00prune_low_magnitude_fc3_prun/clip_by_value_1:z:0*
T0*
_output_shapes
:m
(prune_low_magnitude_fc3_prun/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
&prune_low_magnitude_fc3_prun/truediv_3RealDiv&prune_low_magnitude_fc3_prun/mul_6:z:01prune_low_magnitude_fc3_prun/truediv_3/y:output:0*
T0*
_output_shapes
:i
$prune_low_magnitude_fc3_prun/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc3_prun/mul_7Mul-prune_low_magnitude_fc3_prun/mul_7/x:output:0*prune_low_magnitude_fc3_prun/truediv_3:z:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc3_prun/ReadVariableOp_5ReadVariableOp6prune_low_magnitude_fc3_prun_readvariableop_4_resource*
_output_shapes
:*
dtype0�
"prune_low_magnitude_fc3_prun/Neg_3Neg5prune_low_magnitude_fc3_prun/ReadVariableOp_5:value:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc3_prun/add_7AddV2&prune_low_magnitude_fc3_prun/Neg_3:y:0&prune_low_magnitude_fc3_prun/mul_7:z:0*
T0*
_output_shapes
:i
$prune_low_magnitude_fc3_prun/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc3_prun/mul_8Mul-prune_low_magnitude_fc3_prun/mul_8/x:output:0&prune_low_magnitude_fc3_prun/add_7:z:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc3_prun/StopGradient_3StopGradient&prune_low_magnitude_fc3_prun/mul_8:z:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc3_prun/ReadVariableOp_6ReadVariableOp6prune_low_magnitude_fc3_prun_readvariableop_4_resource*
_output_shapes
:*
dtype0�
"prune_low_magnitude_fc3_prun/add_8AddV25prune_low_magnitude_fc3_prun/ReadVariableOp_6:value:04prune_low_magnitude_fc3_prun/StopGradient_3:output:0*
T0*
_output_shapes
:�
$prune_low_magnitude_fc3_prun/BiasAddBiasAdd-prune_low_magnitude_fc3_prun/MatMul:product:0&prune_low_magnitude_fc3_prun/add_8:z:0*
T0*'
_output_shapes
:���������Q
relu3_enc/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :Q
relu3_enc/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :i
relu3_enc/PowPowrelu3_enc/Pow/x:output:0relu3_enc/Pow/y:output:0*
T0*
_output_shapes
: Y
relu3_enc/CastCastrelu3_enc/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: S
relu3_enc/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :S
relu3_enc/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : o
relu3_enc/Pow_1Powrelu3_enc/Pow_1/x:output:0relu3_enc/Pow_1/y:output:0*
T0*
_output_shapes
: ]
relu3_enc/Cast_1Castrelu3_enc/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: T
relu3_enc/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
relu3_enc/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : e
relu3_enc/Cast_2Castrelu3_enc/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
relu3_enc/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@e
relu3_enc/subSubrelu3_enc/Cast_2:y:0relu3_enc/sub/y:output:0*
T0*
_output_shapes
: d
relu3_enc/Pow_2Powrelu3_enc/Const:output:0relu3_enc/sub:z:0*
T0*
_output_shapes
: b
relu3_enc/sub_1Subrelu3_enc/Cast_1:y:0relu3_enc/Pow_2:z:0*
T0*
_output_shapes
: �
relu3_enc/LessEqual	LessEqual-prune_low_magnitude_fc3_prun/BiasAdd:output:0relu3_enc/sub_1:z:0*
T0*'
_output_shapes
:���������w
relu3_enc/ReluRelu-prune_low_magnitude_fc3_prun/BiasAdd:output:0*
T0*'
_output_shapes
:���������v
relu3_enc/ones_like/ShapeShape-prune_low_magnitude_fc3_prun/BiasAdd:output:0*
T0*
_output_shapes
:^
relu3_enc/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
relu3_enc/ones_likeFill"relu3_enc/ones_like/Shape:output:0"relu3_enc/ones_like/Const:output:0*
T0*'
_output_shapes
:���������b
relu3_enc/sub_2Subrelu3_enc/Cast_1:y:0relu3_enc/Pow_2:z:0*
T0*
_output_shapes
: y
relu3_enc/mulMulrelu3_enc/ones_like:output:0relu3_enc/sub_2:z:0*
T0*'
_output_shapes
:����������
relu3_enc/SelectV2SelectV2relu3_enc/LessEqual:z:0relu3_enc/Relu:activations:0relu3_enc/mul:z:0*
T0*'
_output_shapes
:����������
relu3_enc/mul_1Mul-prune_low_magnitude_fc3_prun/BiasAdd:output:0relu3_enc/Cast:y:0*
T0*'
_output_shapes
:���������y
relu3_enc/truedivRealDivrelu3_enc/mul_1:z:0relu3_enc/Cast_1:y:0*
T0*'
_output_shapes
:���������]
relu3_enc/NegNegrelu3_enc/truediv:z:0*
T0*'
_output_shapes
:���������a
relu3_enc/RoundRoundrelu3_enc/truediv:z:0*
T0*'
_output_shapes
:���������p
relu3_enc/addAddV2relu3_enc/Neg:y:0relu3_enc/Round:y:0*
T0*'
_output_shapes
:���������k
relu3_enc/StopGradientStopGradientrelu3_enc/add:z:0*
T0*'
_output_shapes
:����������
relu3_enc/add_1AddV2relu3_enc/truediv:z:0relu3_enc/StopGradient:output:0*
T0*'
_output_shapes
:���������y
relu3_enc/truediv_1RealDivrelu3_enc/add_1:z:0relu3_enc/Cast:y:0*
T0*'
_output_shapes
:���������Z
relu3_enc/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
relu3_enc/truediv_2RealDivrelu3_enc/truediv_2/x:output:0relu3_enc/Cast:y:0*
T0*
_output_shapes
: V
relu3_enc/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
relu3_enc/sub_3Subrelu3_enc/sub_3/x:output:0relu3_enc/truediv_2:z:0*
T0*
_output_shapes
: �
relu3_enc/clip_by_value/MinimumMinimumrelu3_enc/truediv_1:z:0relu3_enc/sub_3:z:0*
T0*'
_output_shapes
:���������^
relu3_enc/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
relu3_enc/clip_by_valueMaximum#relu3_enc/clip_by_value/Minimum:z:0"relu3_enc/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������{
relu3_enc/mul_2Mulrelu3_enc/Cast_1:y:0relu3_enc/clip_by_value:z:0*
T0*'
_output_shapes
:���������e
relu3_enc/Neg_1Negrelu3_enc/SelectV2:output:0*
T0*'
_output_shapes
:���������t
relu3_enc/add_2AddV2relu3_enc/Neg_1:y:0relu3_enc/mul_2:z:0*
T0*'
_output_shapes
:���������V
relu3_enc/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
relu3_enc/mul_3Mulrelu3_enc/mul_3/x:output:0relu3_enc/add_2:z:0*
T0*'
_output_shapes
:���������o
relu3_enc/StopGradient_1StopGradientrelu3_enc/mul_3:z:0*
T0*'
_output_shapes
:����������
relu3_enc/add_3AddV2relu3_enc/SelectV2:output:0!relu3_enc/StopGradient_1:output:0*
T0*'
_output_shapes
:����������
$encoder_output/MatMul/ReadVariableOpReadVariableOp-encoder_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
encoder_output/MatMulMatMulrelu3_enc/add_3:z:0,encoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%encoder_output/BiasAdd/ReadVariableOpReadVariableOp.encoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
encoder_output/BiasAddBiasAddencoder_output/MatMul:product:0-encoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
encoder_output/ReluReluencoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:����������
2prune_low_magnitude_fc4_prunedclass/ReadVariableOpReadVariableOp;prune_low_magnitude_fc4_prunedclass_readvariableop_resource*
_output_shapes
: *
dtype0	k
)prune_low_magnitude_fc4_prunedclass/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
'prune_low_magnitude_fc4_prunedclass/addAddV2:prune_low_magnitude_fc4_prunedclass/ReadVariableOp:value:02prune_low_magnitude_fc4_prunedclass/add/y:output:0*
T0	*
_output_shapes
: �
4prune_low_magnitude_fc4_prunedclass/AssignVariableOpAssignVariableOp;prune_low_magnitude_fc4_prunedclass_readvariableop_resource+prune_low_magnitude_fc4_prunedclass/add:z:03^prune_low_magnitude_fc4_prunedclass/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(
*prune_low_magnitude_fc4_prunedclass/updateNoOp5^prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
_output_shapes
 �
Gprune_low_magnitude_fc4_prunedclass/assert_greater_equal/ReadVariableOpReadVariableOp;prune_low_magnitude_fc4_prunedclass_readvariableop_resource5^prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
_output_shapes
: *
dtype0	|
:prune_low_magnitude_fc4_prunedclass/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
Eprune_low_magnitude_fc4_prunedclass/assert_greater_equal/GreaterEqualGreaterEqualOprune_low_magnitude_fc4_prunedclass/assert_greater_equal/ReadVariableOp:value:0Cprune_low_magnitude_fc4_prunedclass/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 
=prune_low_magnitude_fc4_prunedclass/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Dprune_low_magnitude_fc4_prunedclass/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Dprune_low_magnitude_fc4_prunedclass/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
>prune_low_magnitude_fc4_prunedclass/assert_greater_equal/rangeRangeMprune_low_magnitude_fc4_prunedclass/assert_greater_equal/range/start:output:0Fprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Rank:output:0Mprune_low_magnitude_fc4_prunedclass/assert_greater_equal/range/delta:output:0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/assert_greater_equal/AllAllIprune_low_magnitude_fc4_prunedclass/assert_greater_equal/GreaterEqual:z:0Gprune_low_magnitude_fc4_prunedclass/assert_greater_equal/range:output:0*
_output_shapes
: �
Eprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Gprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Gprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*a
valueXBV BPx (prune_low_magnitude_fc4_prunedclass/assert_greater_equal/ReadVariableOp:0) = �
Gprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*T
valueKBI BCy (prune_low_magnitude_fc4_prunedclass/assert_greater_equal/y:0) = �
Kprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuardIfEprune_low_magnitude_fc4_prunedclass/assert_greater_equal/All:output:0Eprune_low_magnitude_fc4_prunedclass/assert_greater_equal/All:output:0Oprune_low_magnitude_fc4_prunedclass/assert_greater_equal/ReadVariableOp:value:0Cprune_low_magnitude_fc4_prunedclass/assert_greater_equal/y:output:0E^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *l
else_branch]R[
Yprune_low_magnitude_fc4_prunedclass_assert_greater_equal_Assert_AssertGuard_false_1598230*
output_shapes
: *k
then_branch\RZ
Xprune_low_magnitude_fc4_prunedclass_assert_greater_equal_Assert_AssertGuard_true_1598229�
Tprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/IdentityIdentityTprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
?prune_low_magnitude_fc4_prunedclass/GreaterEqual/ReadVariableOpReadVariableOp;prune_low_magnitude_fc4_prunedclass_readvariableop_resource5^prune_low_magnitude_fc4_prunedclass/AssignVariableOpU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
2prune_low_magnitude_fc4_prunedclass/GreaterEqual/yConstU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
0prune_low_magnitude_fc4_prunedclass/GreaterEqualGreaterEqualGprune_low_magnitude_fc4_prunedclass/GreaterEqual/ReadVariableOp:value:0;prune_low_magnitude_fc4_prunedclass/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/LessEqual/ReadVariableOpReadVariableOp;prune_low_magnitude_fc4_prunedclass_readvariableop_resource5^prune_low_magnitude_fc4_prunedclass/AssignVariableOpU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
/prune_low_magnitude_fc4_prunedclass/LessEqual/yConstU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�N�
-prune_low_magnitude_fc4_prunedclass/LessEqual	LessEqualDprune_low_magnitude_fc4_prunedclass/LessEqual/ReadVariableOp:value:08prune_low_magnitude_fc4_prunedclass/LessEqual/y:output:0*
T0	*
_output_shapes
: �
*prune_low_magnitude_fc4_prunedclass/Less/xConstU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N�
*prune_low_magnitude_fc4_prunedclass/Less/yConstU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : �
(prune_low_magnitude_fc4_prunedclass/LessLess3prune_low_magnitude_fc4_prunedclass/Less/x:output:03prune_low_magnitude_fc4_prunedclass/Less/y:output:0*
T0*
_output_shapes
: �
-prune_low_magnitude_fc4_prunedclass/LogicalOr	LogicalOr1prune_low_magnitude_fc4_prunedclass/LessEqual:z:0,prune_low_magnitude_fc4_prunedclass/Less:z:0*
_output_shapes
: �
.prune_low_magnitude_fc4_prunedclass/LogicalAnd
LogicalAnd4prune_low_magnitude_fc4_prunedclass/GreaterEqual:z:01prune_low_magnitude_fc4_prunedclass/LogicalOr:z:0*
_output_shapes
: �
6prune_low_magnitude_fc4_prunedclass/Sub/ReadVariableOpReadVariableOp;prune_low_magnitude_fc4_prunedclass_readvariableop_resource5^prune_low_magnitude_fc4_prunedclass/AssignVariableOpU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
)prune_low_magnitude_fc4_prunedclass/Sub/yConstU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
'prune_low_magnitude_fc4_prunedclass/SubSub>prune_low_magnitude_fc4_prunedclass/Sub/ReadVariableOp:value:02prune_low_magnitude_fc4_prunedclass/Sub/y:output:0*
T0	*
_output_shapes
: �
.prune_low_magnitude_fc4_prunedclass/FloorMod/yConstU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd�
,prune_low_magnitude_fc4_prunedclass/FloorModFloorMod+prune_low_magnitude_fc4_prunedclass/Sub:z:07prune_low_magnitude_fc4_prunedclass/FloorMod/y:output:0*
T0	*
_output_shapes
: �
+prune_low_magnitude_fc4_prunedclass/Equal/yConstU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R �
)prune_low_magnitude_fc4_prunedclass/EqualEqual0prune_low_magnitude_fc4_prunedclass/FloorMod:z:04prune_low_magnitude_fc4_prunedclass/Equal/y:output:0*
T0	*
_output_shapes
: �
0prune_low_magnitude_fc4_prunedclass/LogicalAnd_1
LogicalAnd2prune_low_magnitude_fc4_prunedclass/LogicalAnd:z:0-prune_low_magnitude_fc4_prunedclass/Equal:z:0*
_output_shapes
: �
)prune_low_magnitude_fc4_prunedclass/ConstConstU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��L?�
(prune_low_magnitude_fc4_prunedclass/condIf4prune_low_magnitude_fc4_prunedclass/LogicalAnd_1:z:0;prune_low_magnitude_fc4_prunedclass_readvariableop_resource0prune_low_magnitude_fc4_prunedclass_cond_input_10prune_low_magnitude_fc4_prunedclass_cond_input_20prune_low_magnitude_fc4_prunedclass_cond_input_34prune_low_magnitude_fc4_prunedclass/LogicalAnd_1:z:05^prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*I
else_branch:R8
6prune_low_magnitude_fc4_prunedclass_cond_false_1598270*
output_shapes
: *H
then_branch9R7
5prune_low_magnitude_fc4_prunedclass_cond_true_1598269�
1prune_low_magnitude_fc4_prunedclass/cond/IdentityIdentity1prune_low_magnitude_fc4_prunedclass/cond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
,prune_low_magnitude_fc4_prunedclass/update_1NoOpU^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity2^prune_low_magnitude_fc4_prunedclass/cond/Identity*
_output_shapes
 �
6prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOpReadVariableOp0prune_low_magnitude_fc4_prunedclass_cond_input_1*
_output_shapes

:*
dtype0�
8prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1ReadVariableOp0prune_low_magnitude_fc4_prunedclass_cond_input_2)^prune_low_magnitude_fc4_prunedclass/cond*
_output_shapes

:*
dtype0�
'prune_low_magnitude_fc4_prunedclass/MulMul>prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp:value:0@prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
6prune_low_magnitude_fc4_prunedclass/AssignVariableOp_1AssignVariableOp0prune_low_magnitude_fc4_prunedclass_cond_input_1+prune_low_magnitude_fc4_prunedclass/Mul:z:07^prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp)^prune_low_magnitude_fc4_prunedclass/cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
.prune_low_magnitude_fc4_prunedclass/group_depsNoOp7^prune_low_magnitude_fc4_prunedclass/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0prune_low_magnitude_fc4_prunedclass/group_deps_1NoOp/^prune_low_magnitude_fc4_prunedclass/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 k
)prune_low_magnitude_fc4_prunedclass/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :k
)prune_low_magnitude_fc4_prunedclass/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
'prune_low_magnitude_fc4_prunedclass/PowPow2prune_low_magnitude_fc4_prunedclass/Pow/x:output:02prune_low_magnitude_fc4_prunedclass/Pow/y:output:0*
T0*
_output_shapes
: �
(prune_low_magnitude_fc4_prunedclass/CastCast+prune_low_magnitude_fc4_prunedclass/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1ReadVariableOp0prune_low_magnitude_fc4_prunedclass_cond_input_17^prune_low_magnitude_fc4_prunedclass/AssignVariableOp_1*
_output_shapes

:*
dtype0p
+prune_low_magnitude_fc4_prunedclass/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
)prune_low_magnitude_fc4_prunedclass/mul_1Mul<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1:value:04prune_low_magnitude_fc4_prunedclass/mul_1/y:output:0*
T0*
_output_shapes

:�
+prune_low_magnitude_fc4_prunedclass/truedivRealDiv-prune_low_magnitude_fc4_prunedclass/mul_1:z:0,prune_low_magnitude_fc4_prunedclass/Cast:y:0*
T0*
_output_shapes

:�
'prune_low_magnitude_fc4_prunedclass/NegNeg/prune_low_magnitude_fc4_prunedclass/truediv:z:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc4_prunedclass/RoundRound/prune_low_magnitude_fc4_prunedclass/truediv:z:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc4_prunedclass/add_1AddV2+prune_low_magnitude_fc4_prunedclass/Neg:y:0-prune_low_magnitude_fc4_prunedclass/Round:y:0*
T0*
_output_shapes

:�
0prune_low_magnitude_fc4_prunedclass/StopGradientStopGradient-prune_low_magnitude_fc4_prunedclass/add_1:z:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc4_prunedclass/add_2AddV2/prune_low_magnitude_fc4_prunedclass/truediv:z:09prune_low_magnitude_fc4_prunedclass/StopGradient:output:0*
T0*
_output_shapes

:�
;prune_low_magnitude_fc4_prunedclass/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
9prune_low_magnitude_fc4_prunedclass/clip_by_value/MinimumMinimum-prune_low_magnitude_fc4_prunedclass/add_2:z:0Dprune_low_magnitude_fc4_prunedclass/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:x
3prune_low_magnitude_fc4_prunedclass/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
1prune_low_magnitude_fc4_prunedclass/clip_by_valueMaximum=prune_low_magnitude_fc4_prunedclass/clip_by_value/Minimum:z:0<prune_low_magnitude_fc4_prunedclass/clip_by_value/y:output:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc4_prunedclass/mul_2Mul,prune_low_magnitude_fc4_prunedclass/Cast:y:05prune_low_magnitude_fc4_prunedclass/clip_by_value:z:0*
T0*
_output_shapes

:t
/prune_low_magnitude_fc4_prunedclass/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
-prune_low_magnitude_fc4_prunedclass/truediv_1RealDiv-prune_low_magnitude_fc4_prunedclass/mul_2:z:08prune_low_magnitude_fc4_prunedclass/truediv_1/y:output:0*
T0*
_output_shapes

:p
+prune_low_magnitude_fc4_prunedclass/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)prune_low_magnitude_fc4_prunedclass/mul_3Mul4prune_low_magnitude_fc4_prunedclass/mul_3/x:output:01prune_low_magnitude_fc4_prunedclass/truediv_1:z:0*
T0*
_output_shapes

:�
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2ReadVariableOp0prune_low_magnitude_fc4_prunedclass_cond_input_17^prune_low_magnitude_fc4_prunedclass/AssignVariableOp_1*
_output_shapes

:*
dtype0�
)prune_low_magnitude_fc4_prunedclass/Neg_1Neg<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2:value:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc4_prunedclass/add_3AddV2-prune_low_magnitude_fc4_prunedclass/Neg_1:y:0-prune_low_magnitude_fc4_prunedclass/mul_3:z:0*
T0*
_output_shapes

:p
+prune_low_magnitude_fc4_prunedclass/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)prune_low_magnitude_fc4_prunedclass/mul_4Mul4prune_low_magnitude_fc4_prunedclass/mul_4/x:output:0-prune_low_magnitude_fc4_prunedclass/add_3:z:0*
T0*
_output_shapes

:�
2prune_low_magnitude_fc4_prunedclass/StopGradient_1StopGradient-prune_low_magnitude_fc4_prunedclass/mul_4:z:0*
T0*
_output_shapes

:�
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3ReadVariableOp0prune_low_magnitude_fc4_prunedclass_cond_input_17^prune_low_magnitude_fc4_prunedclass/AssignVariableOp_1*
_output_shapes

:*
dtype0�
)prune_low_magnitude_fc4_prunedclass/add_4AddV2<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3:value:0;prune_low_magnitude_fc4_prunedclass/StopGradient_1:output:0*
T0*
_output_shapes

:�
*prune_low_magnitude_fc4_prunedclass/MatMulMatMul!encoder_output/Relu:activations:0-prune_low_magnitude_fc4_prunedclass/add_4:z:0*
T0*'
_output_shapes
:���������m
+prune_low_magnitude_fc4_prunedclass/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :m
+prune_low_magnitude_fc4_prunedclass/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
)prune_low_magnitude_fc4_prunedclass/Pow_1Pow4prune_low_magnitude_fc4_prunedclass/Pow_1/x:output:04prune_low_magnitude_fc4_prunedclass/Pow_1/y:output:0*
T0*
_output_shapes
: �
*prune_low_magnitude_fc4_prunedclass/Cast_1Cast-prune_low_magnitude_fc4_prunedclass/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4ReadVariableOp=prune_low_magnitude_fc4_prunedclass_readvariableop_4_resource*
_output_shapes
:*
dtype0p
+prune_low_magnitude_fc4_prunedclass/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
)prune_low_magnitude_fc4_prunedclass/mul_5Mul<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4:value:04prune_low_magnitude_fc4_prunedclass/mul_5/y:output:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc4_prunedclass/truediv_2RealDiv-prune_low_magnitude_fc4_prunedclass/mul_5:z:0.prune_low_magnitude_fc4_prunedclass/Cast_1:y:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/Neg_2Neg1prune_low_magnitude_fc4_prunedclass/truediv_2:z:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc4_prunedclass/Round_1Round1prune_low_magnitude_fc4_prunedclass/truediv_2:z:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/add_5AddV2-prune_low_magnitude_fc4_prunedclass/Neg_2:y:0/prune_low_magnitude_fc4_prunedclass/Round_1:y:0*
T0*
_output_shapes
:�
2prune_low_magnitude_fc4_prunedclass/StopGradient_2StopGradient-prune_low_magnitude_fc4_prunedclass/add_5:z:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/add_6AddV21prune_low_magnitude_fc4_prunedclass/truediv_2:z:0;prune_low_magnitude_fc4_prunedclass/StopGradient_2:output:0*
T0*
_output_shapes
:�
=prune_low_magnitude_fc4_prunedclass/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
;prune_low_magnitude_fc4_prunedclass/clip_by_value_1/MinimumMinimum-prune_low_magnitude_fc4_prunedclass/add_6:z:0Fprune_low_magnitude_fc4_prunedclass/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:z
5prune_low_magnitude_fc4_prunedclass/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
3prune_low_magnitude_fc4_prunedclass/clip_by_value_1Maximum?prune_low_magnitude_fc4_prunedclass/clip_by_value_1/Minimum:z:0>prune_low_magnitude_fc4_prunedclass/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/mul_6Mul.prune_low_magnitude_fc4_prunedclass/Cast_1:y:07prune_low_magnitude_fc4_prunedclass/clip_by_value_1:z:0*
T0*
_output_shapes
:t
/prune_low_magnitude_fc4_prunedclass/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
-prune_low_magnitude_fc4_prunedclass/truediv_3RealDiv-prune_low_magnitude_fc4_prunedclass/mul_6:z:08prune_low_magnitude_fc4_prunedclass/truediv_3/y:output:0*
T0*
_output_shapes
:p
+prune_low_magnitude_fc4_prunedclass/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)prune_low_magnitude_fc4_prunedclass/mul_7Mul4prune_low_magnitude_fc4_prunedclass/mul_7/x:output:01prune_low_magnitude_fc4_prunedclass/truediv_3:z:0*
T0*
_output_shapes
:�
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5ReadVariableOp=prune_low_magnitude_fc4_prunedclass_readvariableop_4_resource*
_output_shapes
:*
dtype0�
)prune_low_magnitude_fc4_prunedclass/Neg_3Neg<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5:value:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/add_7AddV2-prune_low_magnitude_fc4_prunedclass/Neg_3:y:0-prune_low_magnitude_fc4_prunedclass/mul_7:z:0*
T0*
_output_shapes
:p
+prune_low_magnitude_fc4_prunedclass/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)prune_low_magnitude_fc4_prunedclass/mul_8Mul4prune_low_magnitude_fc4_prunedclass/mul_8/x:output:0-prune_low_magnitude_fc4_prunedclass/add_7:z:0*
T0*
_output_shapes
:�
2prune_low_magnitude_fc4_prunedclass/StopGradient_3StopGradient-prune_low_magnitude_fc4_prunedclass/mul_8:z:0*
T0*
_output_shapes
:�
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_6ReadVariableOp=prune_low_magnitude_fc4_prunedclass_readvariableop_4_resource*
_output_shapes
:*
dtype0�
)prune_low_magnitude_fc4_prunedclass/add_8AddV2<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_6:value:0;prune_low_magnitude_fc4_prunedclass/StopGradient_3:output:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc4_prunedclass/BiasAddBiasAdd4prune_low_magnitude_fc4_prunedclass/MatMul:product:0-prune_low_magnitude_fc4_prunedclass/add_8:z:0*
T0*'
_output_shapes
:���������W
prunclass_relu4/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :W
prunclass_relu4/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :{
prunclass_relu4/PowPowprunclass_relu4/Pow/x:output:0prunclass_relu4/Pow/y:output:0*
T0*
_output_shapes
: e
prunclass_relu4/CastCastprunclass_relu4/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: Y
prunclass_relu4/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
prunclass_relu4/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
prunclass_relu4/Pow_1Pow prunclass_relu4/Pow_1/x:output:0 prunclass_relu4/Pow_1/y:output:0*
T0*
_output_shapes
: i
prunclass_relu4/Cast_1Castprunclass_relu4/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
prunclass_relu4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
prunclass_relu4/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : q
prunclass_relu4/Cast_2Cast!prunclass_relu4/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
prunclass_relu4/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
prunclass_relu4/subSubprunclass_relu4/Cast_2:y:0prunclass_relu4/sub/y:output:0*
T0*
_output_shapes
: v
prunclass_relu4/Pow_2Powprunclass_relu4/Const:output:0prunclass_relu4/sub:z:0*
T0*
_output_shapes
: t
prunclass_relu4/sub_1Subprunclass_relu4/Cast_1:y:0prunclass_relu4/Pow_2:z:0*
T0*
_output_shapes
: �
prunclass_relu4/LessEqual	LessEqual4prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0prunclass_relu4/sub_1:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/ReluRelu4prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0*
T0*'
_output_shapes
:����������
prunclass_relu4/ones_like/ShapeShape4prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0*
T0*
_output_shapes
:d
prunclass_relu4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
prunclass_relu4/ones_likeFill(prunclass_relu4/ones_like/Shape:output:0(prunclass_relu4/ones_like/Const:output:0*
T0*'
_output_shapes
:���������t
prunclass_relu4/sub_2Subprunclass_relu4/Cast_1:y:0prunclass_relu4/Pow_2:z:0*
T0*
_output_shapes
: �
prunclass_relu4/mulMul"prunclass_relu4/ones_like:output:0prunclass_relu4/sub_2:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/SelectV2SelectV2prunclass_relu4/LessEqual:z:0"prunclass_relu4/Relu:activations:0prunclass_relu4/mul:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/mul_1Mul4prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0prunclass_relu4/Cast:y:0*
T0*'
_output_shapes
:����������
prunclass_relu4/truedivRealDivprunclass_relu4/mul_1:z:0prunclass_relu4/Cast_1:y:0*
T0*'
_output_shapes
:���������i
prunclass_relu4/NegNegprunclass_relu4/truediv:z:0*
T0*'
_output_shapes
:���������m
prunclass_relu4/RoundRoundprunclass_relu4/truediv:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/addAddV2prunclass_relu4/Neg:y:0prunclass_relu4/Round:y:0*
T0*'
_output_shapes
:���������w
prunclass_relu4/StopGradientStopGradientprunclass_relu4/add:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/add_1AddV2prunclass_relu4/truediv:z:0%prunclass_relu4/StopGradient:output:0*
T0*'
_output_shapes
:����������
prunclass_relu4/truediv_1RealDivprunclass_relu4/add_1:z:0prunclass_relu4/Cast:y:0*
T0*'
_output_shapes
:���������`
prunclass_relu4/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
prunclass_relu4/truediv_2RealDiv$prunclass_relu4/truediv_2/x:output:0prunclass_relu4/Cast:y:0*
T0*
_output_shapes
: \
prunclass_relu4/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
prunclass_relu4/sub_3Sub prunclass_relu4/sub_3/x:output:0prunclass_relu4/truediv_2:z:0*
T0*
_output_shapes
: �
%prunclass_relu4/clip_by_value/MinimumMinimumprunclass_relu4/truediv_1:z:0prunclass_relu4/sub_3:z:0*
T0*'
_output_shapes
:���������d
prunclass_relu4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
prunclass_relu4/clip_by_valueMaximum)prunclass_relu4/clip_by_value/Minimum:z:0(prunclass_relu4/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
prunclass_relu4/mul_2Mulprunclass_relu4/Cast_1:y:0!prunclass_relu4/clip_by_value:z:0*
T0*'
_output_shapes
:���������q
prunclass_relu4/Neg_1Neg!prunclass_relu4/SelectV2:output:0*
T0*'
_output_shapes
:����������
prunclass_relu4/add_2AddV2prunclass_relu4/Neg_1:y:0prunclass_relu4/mul_2:z:0*
T0*'
_output_shapes
:���������\
prunclass_relu4/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
prunclass_relu4/mul_3Mul prunclass_relu4/mul_3/x:output:0prunclass_relu4/add_2:z:0*
T0*'
_output_shapes
:���������{
prunclass_relu4/StopGradient_1StopGradientprunclass_relu4/mul_3:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/add_3AddV2!prunclass_relu4/SelectV2:output:0'prunclass_relu4/StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
fc5_class/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :Q
fc5_class/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : i
fc5_class/PowPowfc5_class/Pow/x:output:0fc5_class/Pow/y:output:0*
T0*
_output_shapes
: Y
fc5_class/CastCastfc5_class/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: z
fc5_class/ReadVariableOpReadVariableOp!fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0T
fc5_class/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   By
fc5_class/mulMul fc5_class/ReadVariableOp:value:0fc5_class/mul/y:output:0*
T0*
_output_shapes

:(l
fc5_class/truedivRealDivfc5_class/mul:z:0fc5_class/Cast:y:0*
T0*
_output_shapes

:(T
fc5_class/NegNegfc5_class/truediv:z:0*
T0*
_output_shapes

:(X
fc5_class/RoundRoundfc5_class/truediv:z:0*
T0*
_output_shapes

:(g
fc5_class/addAddV2fc5_class/Neg:y:0fc5_class/Round:y:0*
T0*
_output_shapes

:(b
fc5_class/StopGradientStopGradientfc5_class/add:z:0*
T0*
_output_shapes

:(y
fc5_class/add_1AddV2fc5_class/truediv:z:0fc5_class/StopGradient:output:0*
T0*
_output_shapes

:(f
!fc5_class/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
fc5_class/clip_by_value/MinimumMinimumfc5_class/add_1:z:0*fc5_class/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(^
fc5_class/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
fc5_class/clip_by_valueMaximum#fc5_class/clip_by_value/Minimum:z:0"fc5_class/clip_by_value/y:output:0*
T0*
_output_shapes

:(p
fc5_class/mul_1Mulfc5_class/Cast:y:0fc5_class/clip_by_value:z:0*
T0*
_output_shapes

:(Z
fc5_class/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B|
fc5_class/truediv_1RealDivfc5_class/mul_1:z:0fc5_class/truediv_1/y:output:0*
T0*
_output_shapes

:(V
fc5_class/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
fc5_class/mul_2Mulfc5_class/mul_2/x:output:0fc5_class/truediv_1:z:0*
T0*
_output_shapes

:(|
fc5_class/ReadVariableOp_1ReadVariableOp!fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0c
fc5_class/Neg_1Neg"fc5_class/ReadVariableOp_1:value:0*
T0*
_output_shapes

:(k
fc5_class/add_2AddV2fc5_class/Neg_1:y:0fc5_class/mul_2:z:0*
T0*
_output_shapes

:(V
fc5_class/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?p
fc5_class/mul_3Mulfc5_class/mul_3/x:output:0fc5_class/add_2:z:0*
T0*
_output_shapes

:(f
fc5_class/StopGradient_1StopGradientfc5_class/mul_3:z:0*
T0*
_output_shapes

:(|
fc5_class/ReadVariableOp_2ReadVariableOp!fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0�
fc5_class/add_3AddV2"fc5_class/ReadVariableOp_2:value:0!fc5_class/StopGradient_1:output:0*
T0*
_output_shapes

:(|
fc5_class/MatMulMatMulprunclass_relu4/add_3:z:0fc5_class/add_3:z:0*
T0*'
_output_shapes
:���������(S
fc5_class/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :S
fc5_class/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : o
fc5_class/Pow_1Powfc5_class/Pow_1/x:output:0fc5_class/Pow_1/y:output:0*
T0*
_output_shapes
: ]
fc5_class/Cast_1Castfc5_class/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: z
fc5_class/ReadVariableOp_3ReadVariableOp#fc5_class_readvariableop_3_resource*
_output_shapes
:(*
dtype0V
fc5_class/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B{
fc5_class/mul_4Mul"fc5_class/ReadVariableOp_3:value:0fc5_class/mul_4/y:output:0*
T0*
_output_shapes
:(n
fc5_class/truediv_2RealDivfc5_class/mul_4:z:0fc5_class/Cast_1:y:0*
T0*
_output_shapes
:(T
fc5_class/Neg_2Negfc5_class/truediv_2:z:0*
T0*
_output_shapes
:(X
fc5_class/Round_1Roundfc5_class/truediv_2:z:0*
T0*
_output_shapes
:(i
fc5_class/add_4AddV2fc5_class/Neg_2:y:0fc5_class/Round_1:y:0*
T0*
_output_shapes
:(b
fc5_class/StopGradient_2StopGradientfc5_class/add_4:z:0*
T0*
_output_shapes
:(y
fc5_class/add_5AddV2fc5_class/truediv_2:z:0!fc5_class/StopGradient_2:output:0*
T0*
_output_shapes
:(h
#fc5_class/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
!fc5_class/clip_by_value_1/MinimumMinimumfc5_class/add_5:z:0,fc5_class/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(`
fc5_class/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
fc5_class/clip_by_value_1Maximum%fc5_class/clip_by_value_1/Minimum:z:0$fc5_class/clip_by_value_1/y:output:0*
T0*
_output_shapes
:(p
fc5_class/mul_5Mulfc5_class/Cast_1:y:0fc5_class/clip_by_value_1:z:0*
T0*
_output_shapes
:(Z
fc5_class/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bx
fc5_class/truediv_3RealDivfc5_class/mul_5:z:0fc5_class/truediv_3/y:output:0*
T0*
_output_shapes
:(V
fc5_class/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?p
fc5_class/mul_6Mulfc5_class/mul_6/x:output:0fc5_class/truediv_3:z:0*
T0*
_output_shapes
:(z
fc5_class/ReadVariableOp_4ReadVariableOp#fc5_class_readvariableop_3_resource*
_output_shapes
:(*
dtype0_
fc5_class/Neg_3Neg"fc5_class/ReadVariableOp_4:value:0*
T0*
_output_shapes
:(g
fc5_class/add_6AddV2fc5_class/Neg_3:y:0fc5_class/mul_6:z:0*
T0*
_output_shapes
:(V
fc5_class/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
fc5_class/mul_7Mulfc5_class/mul_7/x:output:0fc5_class/add_6:z:0*
T0*
_output_shapes
:(b
fc5_class/StopGradient_3StopGradientfc5_class/mul_7:z:0*
T0*
_output_shapes
:(z
fc5_class/ReadVariableOp_5ReadVariableOp#fc5_class_readvariableop_3_resource*
_output_shapes
:(*
dtype0�
fc5_class/add_7AddV2"fc5_class/ReadVariableOp_5:value:0!fc5_class/StopGradient_3:output:0*
T0*
_output_shapes
:(
fc5_class/BiasAddBiasAddfc5_class/MatMul:product:0fc5_class/add_7:z:0*
T0*'
_output_shapes
:���������(S
class_relu5/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :S
class_relu5/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :o
class_relu5/PowPowclass_relu5/Pow/x:output:0class_relu5/Pow/y:output:0*
T0*
_output_shapes
: ]
class_relu5/CastCastclass_relu5/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: U
class_relu5/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :U
class_relu5/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : u
class_relu5/Pow_1Powclass_relu5/Pow_1/x:output:0class_relu5/Pow_1/y:output:0*
T0*
_output_shapes
: a
class_relu5/Cast_1Castclass_relu5/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: V
class_relu5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @V
class_relu5/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : i
class_relu5/Cast_2Castclass_relu5/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: V
class_relu5/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@k
class_relu5/subSubclass_relu5/Cast_2:y:0class_relu5/sub/y:output:0*
T0*
_output_shapes
: j
class_relu5/Pow_2Powclass_relu5/Const:output:0class_relu5/sub:z:0*
T0*
_output_shapes
: h
class_relu5/sub_1Subclass_relu5/Cast_1:y:0class_relu5/Pow_2:z:0*
T0*
_output_shapes
: �
class_relu5/LessEqual	LessEqualfc5_class/BiasAdd:output:0class_relu5/sub_1:z:0*
T0*'
_output_shapes
:���������(f
class_relu5/ReluRelufc5_class/BiasAdd:output:0*
T0*'
_output_shapes
:���������(e
class_relu5/ones_like/ShapeShapefc5_class/BiasAdd:output:0*
T0*
_output_shapes
:`
class_relu5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
class_relu5/ones_likeFill$class_relu5/ones_like/Shape:output:0$class_relu5/ones_like/Const:output:0*
T0*'
_output_shapes
:���������(h
class_relu5/sub_2Subclass_relu5/Cast_1:y:0class_relu5/Pow_2:z:0*
T0*
_output_shapes
: 
class_relu5/mulMulclass_relu5/ones_like:output:0class_relu5/sub_2:z:0*
T0*'
_output_shapes
:���������(�
class_relu5/SelectV2SelectV2class_relu5/LessEqual:z:0class_relu5/Relu:activations:0class_relu5/mul:z:0*
T0*'
_output_shapes
:���������(|
class_relu5/mul_1Mulfc5_class/BiasAdd:output:0class_relu5/Cast:y:0*
T0*'
_output_shapes
:���������(
class_relu5/truedivRealDivclass_relu5/mul_1:z:0class_relu5/Cast_1:y:0*
T0*'
_output_shapes
:���������(a
class_relu5/NegNegclass_relu5/truediv:z:0*
T0*'
_output_shapes
:���������(e
class_relu5/RoundRoundclass_relu5/truediv:z:0*
T0*'
_output_shapes
:���������(v
class_relu5/addAddV2class_relu5/Neg:y:0class_relu5/Round:y:0*
T0*'
_output_shapes
:���������(o
class_relu5/StopGradientStopGradientclass_relu5/add:z:0*
T0*'
_output_shapes
:���������(�
class_relu5/add_1AddV2class_relu5/truediv:z:0!class_relu5/StopGradient:output:0*
T0*'
_output_shapes
:���������(
class_relu5/truediv_1RealDivclass_relu5/add_1:z:0class_relu5/Cast:y:0*
T0*'
_output_shapes
:���������(\
class_relu5/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
class_relu5/truediv_2RealDiv class_relu5/truediv_2/x:output:0class_relu5/Cast:y:0*
T0*
_output_shapes
: X
class_relu5/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?r
class_relu5/sub_3Subclass_relu5/sub_3/x:output:0class_relu5/truediv_2:z:0*
T0*
_output_shapes
: �
!class_relu5/clip_by_value/MinimumMinimumclass_relu5/truediv_1:z:0class_relu5/sub_3:z:0*
T0*'
_output_shapes
:���������(`
class_relu5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
class_relu5/clip_by_valueMaximum%class_relu5/clip_by_value/Minimum:z:0$class_relu5/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������(�
class_relu5/mul_2Mulclass_relu5/Cast_1:y:0class_relu5/clip_by_value:z:0*
T0*'
_output_shapes
:���������(i
class_relu5/Neg_1Negclass_relu5/SelectV2:output:0*
T0*'
_output_shapes
:���������(z
class_relu5/add_2AddV2class_relu5/Neg_1:y:0class_relu5/mul_2:z:0*
T0*'
_output_shapes
:���������(X
class_relu5/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
class_relu5/mul_3Mulclass_relu5/mul_3/x:output:0class_relu5/add_2:z:0*
T0*'
_output_shapes
:���������(s
class_relu5/StopGradient_1StopGradientclass_relu5/mul_3:z:0*
T0*'
_output_shapes
:���������(�
class_relu5/add_3AddV2class_relu5/SelectV2:output:0#class_relu5/StopGradient_1:output:0*
T0*'
_output_shapes
:���������(V
classifier_out/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :V
classifier_out/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :x
classifier_out/PowPowclassifier_out/Pow/x:output:0classifier_out/Pow/y:output:0*
T0*
_output_shapes
: c
classifier_out/CastCastclassifier_out/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
classifier_out/ReadVariableOpReadVariableOp&classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0Y
classifier_out/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier_out/mulMul%classifier_out/ReadVariableOp:value:0classifier_out/mul/y:output:0*
T0*
_output_shapes

:(
{
classifier_out/truedivRealDivclassifier_out/mul:z:0classifier_out/Cast:y:0*
T0*
_output_shapes

:(
^
classifier_out/NegNegclassifier_out/truediv:z:0*
T0*
_output_shapes

:(
b
classifier_out/RoundRoundclassifier_out/truediv:z:0*
T0*
_output_shapes

:(
v
classifier_out/addAddV2classifier_out/Neg:y:0classifier_out/Round:y:0*
T0*
_output_shapes

:(
l
classifier_out/StopGradientStopGradientclassifier_out/add:z:0*
T0*
_output_shapes

:(
�
classifier_out/add_1AddV2classifier_out/truediv:z:0$classifier_out/StopGradient:output:0*
T0*
_output_shapes

:(
k
&classifier_out/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��F�
$classifier_out/clip_by_value/MinimumMinimumclassifier_out/add_1:z:0/classifier_out/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(
c
classifier_out/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
classifier_out/clip_by_valueMaximum(classifier_out/clip_by_value/Minimum:z:0'classifier_out/clip_by_value/y:output:0*
T0*
_output_shapes

:(

classifier_out/mul_1Mulclassifier_out/Cast:y:0 classifier_out/clip_by_value:z:0*
T0*
_output_shapes

:(
_
classifier_out/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier_out/truediv_1RealDivclassifier_out/mul_1:z:0#classifier_out/truediv_1/y:output:0*
T0*
_output_shapes

:(
[
classifier_out/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier_out/mul_2Mulclassifier_out/mul_2/x:output:0classifier_out/truediv_1:z:0*
T0*
_output_shapes

:(
�
classifier_out/ReadVariableOp_1ReadVariableOp&classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0m
classifier_out/Neg_1Neg'classifier_out/ReadVariableOp_1:value:0*
T0*
_output_shapes

:(
z
classifier_out/add_2AddV2classifier_out/Neg_1:y:0classifier_out/mul_2:z:0*
T0*
_output_shapes

:(
[
classifier_out/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
classifier_out/mul_3Mulclassifier_out/mul_3/x:output:0classifier_out/add_2:z:0*
T0*
_output_shapes

:(
p
classifier_out/StopGradient_1StopGradientclassifier_out/mul_3:z:0*
T0*
_output_shapes

:(
�
classifier_out/ReadVariableOp_2ReadVariableOp&classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0�
classifier_out/add_3AddV2'classifier_out/ReadVariableOp_2:value:0&classifier_out/StopGradient_1:output:0*
T0*
_output_shapes

:(
�
classifier_out/MatMulMatMulclass_relu5/add_3:z:0classifier_out/add_3:z:0*
T0*'
_output_shapes
:���������
X
classifier_out/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :X
classifier_out/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
classifier_out/Pow_1Powclassifier_out/Pow_1/x:output:0classifier_out/Pow_1/y:output:0*
T0*
_output_shapes
: g
classifier_out/Cast_1Castclassifier_out/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
classifier_out/ReadVariableOp_3ReadVariableOp(classifier_out_readvariableop_3_resource*
_output_shapes
:
*
dtype0[
classifier_out/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier_out/mul_4Mul'classifier_out/ReadVariableOp_3:value:0classifier_out/mul_4/y:output:0*
T0*
_output_shapes
:
}
classifier_out/truediv_2RealDivclassifier_out/mul_4:z:0classifier_out/Cast_1:y:0*
T0*
_output_shapes
:
^
classifier_out/Neg_2Negclassifier_out/truediv_2:z:0*
T0*
_output_shapes
:
b
classifier_out/Round_1Roundclassifier_out/truediv_2:z:0*
T0*
_output_shapes
:
x
classifier_out/add_4AddV2classifier_out/Neg_2:y:0classifier_out/Round_1:y:0*
T0*
_output_shapes
:
l
classifier_out/StopGradient_2StopGradientclassifier_out/add_4:z:0*
T0*
_output_shapes
:
�
classifier_out/add_5AddV2classifier_out/truediv_2:z:0&classifier_out/StopGradient_2:output:0*
T0*
_output_shapes
:
m
(classifier_out/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��F�
&classifier_out/clip_by_value_1/MinimumMinimumclassifier_out/add_5:z:01classifier_out/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:
e
 classifier_out/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
classifier_out/clip_by_value_1Maximum*classifier_out/clip_by_value_1/Minimum:z:0)classifier_out/clip_by_value_1/y:output:0*
T0*
_output_shapes
:

classifier_out/mul_5Mulclassifier_out/Cast_1:y:0"classifier_out/clip_by_value_1:z:0*
T0*
_output_shapes
:
_
classifier_out/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier_out/truediv_3RealDivclassifier_out/mul_5:z:0#classifier_out/truediv_3/y:output:0*
T0*
_output_shapes
:
[
classifier_out/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
classifier_out/mul_6Mulclassifier_out/mul_6/x:output:0classifier_out/truediv_3:z:0*
T0*
_output_shapes
:
�
classifier_out/ReadVariableOp_4ReadVariableOp(classifier_out_readvariableop_3_resource*
_output_shapes
:
*
dtype0i
classifier_out/Neg_3Neg'classifier_out/ReadVariableOp_4:value:0*
T0*
_output_shapes
:
v
classifier_out/add_6AddV2classifier_out/Neg_3:y:0classifier_out/mul_6:z:0*
T0*
_output_shapes
:
[
classifier_out/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{
classifier_out/mul_7Mulclassifier_out/mul_7/x:output:0classifier_out/add_6:z:0*
T0*
_output_shapes
:
l
classifier_out/StopGradient_3StopGradientclassifier_out/mul_7:z:0*
T0*
_output_shapes
:
�
classifier_out/ReadVariableOp_5ReadVariableOp(classifier_out_readvariableop_3_resource*
_output_shapes
:
*
dtype0�
classifier_out/add_7AddV2'classifier_out/ReadVariableOp_5:value:0&classifier_out/StopGradient_3:output:0*
T0*
_output_shapes
:
�
classifier_out/BiasAddBiasAddclassifier_out/MatMul:product:0classifier_out/add_7:z:0*
T0*'
_output_shapes
:���������
w
classifier_output/SoftmaxSoftmaxclassifier_out/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0prune_low_magnitude_fc4_prunedclass_cond_input_17^prune_low_magnitude_fc4_prunedclass/AssignVariableOp_1*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
/fc5_class/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0�
 fc5_class/kernel/Regularizer/AbsAbs7fc5_class/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(s
"fc5_class/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 fc5_class/kernel/Regularizer/SumSum$fc5_class/kernel/Regularizer/Abs:y:0+fc5_class/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"fc5_class/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
 fc5_class/kernel/Regularizer/mulMul+fc5_class/kernel/Regularizer/mul/x:output:0)fc5_class/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4classifier_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0�
%classifier_out/kernel/Regularizer/AbsAbs<classifier_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(
x
'classifier_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%classifier_out/kernel/Regularizer/SumSum)classifier_out/kernel/Regularizer/Abs:y:00classifier_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'classifier_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%classifier_out/kernel/Regularizer/mulMul0classifier_out/kernel/Regularizer/mul/x:output:0.classifier_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#classifier_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^classifier_out/ReadVariableOp ^classifier_out/ReadVariableOp_1 ^classifier_out/ReadVariableOp_2 ^classifier_out/ReadVariableOp_3 ^classifier_out/ReadVariableOp_4 ^classifier_out/ReadVariableOp_55^classifier_out/kernel/Regularizer/Abs/ReadVariableOp&^encoder_output/BiasAdd/ReadVariableOp%^encoder_output/MatMul/ReadVariableOp^fc1/ReadVariableOp^fc1/ReadVariableOp_1^fc1/ReadVariableOp_2^fc1/ReadVariableOp_3^fc1/ReadVariableOp_4^fc1/ReadVariableOp_5^fc5_class/ReadVariableOp^fc5_class/ReadVariableOp_1^fc5_class/ReadVariableOp_2^fc5_class/ReadVariableOp_3^fc5_class/ReadVariableOp_4^fc5_class/ReadVariableOp_50^fc5_class/kernel/Regularizer/Abs/ReadVariableOp.^prune_low_magnitude_fc2_prun/AssignVariableOp0^prune_low_magnitude_fc2_prun/AssignVariableOp_19^prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOp6^prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOp0^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1,^prune_low_magnitude_fc2_prun/ReadVariableOp.^prune_low_magnitude_fc2_prun/ReadVariableOp_1.^prune_low_magnitude_fc2_prun/ReadVariableOp_2.^prune_low_magnitude_fc2_prun/ReadVariableOp_3.^prune_low_magnitude_fc2_prun/ReadVariableOp_4.^prune_low_magnitude_fc2_prun/ReadVariableOp_5.^prune_low_magnitude_fc2_prun/ReadVariableOp_60^prune_low_magnitude_fc2_prun/Sub/ReadVariableOpE^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuardA^prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp"^prune_low_magnitude_fc2_prun/cond.^prune_low_magnitude_fc3_prun/AssignVariableOp0^prune_low_magnitude_fc3_prun/AssignVariableOp_19^prune_low_magnitude_fc3_prun/GreaterEqual/ReadVariableOp6^prune_low_magnitude_fc3_prun/LessEqual/ReadVariableOp0^prune_low_magnitude_fc3_prun/Mul/ReadVariableOp2^prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1,^prune_low_magnitude_fc3_prun/ReadVariableOp.^prune_low_magnitude_fc3_prun/ReadVariableOp_1.^prune_low_magnitude_fc3_prun/ReadVariableOp_2.^prune_low_magnitude_fc3_prun/ReadVariableOp_3.^prune_low_magnitude_fc3_prun/ReadVariableOp_4.^prune_low_magnitude_fc3_prun/ReadVariableOp_5.^prune_low_magnitude_fc3_prun/ReadVariableOp_60^prune_low_magnitude_fc3_prun/Sub/ReadVariableOpE^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuardA^prune_low_magnitude_fc3_prun/assert_greater_equal/ReadVariableOp"^prune_low_magnitude_fc3_prun/cond5^prune_low_magnitude_fc4_prunedclass/AssignVariableOp7^prune_low_magnitude_fc4_prunedclass/AssignVariableOp_1@^prune_low_magnitude_fc4_prunedclass/GreaterEqual/ReadVariableOp=^prune_low_magnitude_fc4_prunedclass/LessEqual/ReadVariableOp7^prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp9^prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_13^prune_low_magnitude_fc4_prunedclass/ReadVariableOp5^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_15^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_25^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_35^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_45^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_55^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_67^prune_low_magnitude_fc4_prunedclass/Sub/ReadVariableOpL^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuardH^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/ReadVariableOp)^prune_low_magnitude_fc4_prunedclass/condJ^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������@: : : : : : : : : : : : : : : : : : : : : : : 2>
classifier_out/ReadVariableOpclassifier_out/ReadVariableOp2B
classifier_out/ReadVariableOp_1classifier_out/ReadVariableOp_12B
classifier_out/ReadVariableOp_2classifier_out/ReadVariableOp_22B
classifier_out/ReadVariableOp_3classifier_out/ReadVariableOp_32B
classifier_out/ReadVariableOp_4classifier_out/ReadVariableOp_42B
classifier_out/ReadVariableOp_5classifier_out/ReadVariableOp_52l
4classifier_out/kernel/Regularizer/Abs/ReadVariableOp4classifier_out/kernel/Regularizer/Abs/ReadVariableOp2N
%encoder_output/BiasAdd/ReadVariableOp%encoder_output/BiasAdd/ReadVariableOp2L
$encoder_output/MatMul/ReadVariableOp$encoder_output/MatMul/ReadVariableOp2(
fc1/ReadVariableOpfc1/ReadVariableOp2,
fc1/ReadVariableOp_1fc1/ReadVariableOp_12,
fc1/ReadVariableOp_2fc1/ReadVariableOp_22,
fc1/ReadVariableOp_3fc1/ReadVariableOp_32,
fc1/ReadVariableOp_4fc1/ReadVariableOp_42,
fc1/ReadVariableOp_5fc1/ReadVariableOp_524
fc5_class/ReadVariableOpfc5_class/ReadVariableOp28
fc5_class/ReadVariableOp_1fc5_class/ReadVariableOp_128
fc5_class/ReadVariableOp_2fc5_class/ReadVariableOp_228
fc5_class/ReadVariableOp_3fc5_class/ReadVariableOp_328
fc5_class/ReadVariableOp_4fc5_class/ReadVariableOp_428
fc5_class/ReadVariableOp_5fc5_class/ReadVariableOp_52b
/fc5_class/kernel/Regularizer/Abs/ReadVariableOp/fc5_class/kernel/Regularizer/Abs/ReadVariableOp2^
-prune_low_magnitude_fc2_prun/AssignVariableOp-prune_low_magnitude_fc2_prun/AssignVariableOp2b
/prune_low_magnitude_fc2_prun/AssignVariableOp_1/prune_low_magnitude_fc2_prun/AssignVariableOp_12t
8prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOp8prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOp2n
5prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOp5prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOp2b
/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2f
1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_11prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_12Z
+prune_low_magnitude_fc2_prun/ReadVariableOp+prune_low_magnitude_fc2_prun/ReadVariableOp2^
-prune_low_magnitude_fc2_prun/ReadVariableOp_1-prune_low_magnitude_fc2_prun/ReadVariableOp_12^
-prune_low_magnitude_fc2_prun/ReadVariableOp_2-prune_low_magnitude_fc2_prun/ReadVariableOp_22^
-prune_low_magnitude_fc2_prun/ReadVariableOp_3-prune_low_magnitude_fc2_prun/ReadVariableOp_32^
-prune_low_magnitude_fc2_prun/ReadVariableOp_4-prune_low_magnitude_fc2_prun/ReadVariableOp_42^
-prune_low_magnitude_fc2_prun/ReadVariableOp_5-prune_low_magnitude_fc2_prun/ReadVariableOp_52^
-prune_low_magnitude_fc2_prun/ReadVariableOp_6-prune_low_magnitude_fc2_prun/ReadVariableOp_62b
/prune_low_magnitude_fc2_prun/Sub/ReadVariableOp/prune_low_magnitude_fc2_prun/Sub/ReadVariableOp2�
Dprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuardDprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard2�
@prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp@prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp2F
!prune_low_magnitude_fc2_prun/cond!prune_low_magnitude_fc2_prun/cond2^
-prune_low_magnitude_fc3_prun/AssignVariableOp-prune_low_magnitude_fc3_prun/AssignVariableOp2b
/prune_low_magnitude_fc3_prun/AssignVariableOp_1/prune_low_magnitude_fc3_prun/AssignVariableOp_12t
8prune_low_magnitude_fc3_prun/GreaterEqual/ReadVariableOp8prune_low_magnitude_fc3_prun/GreaterEqual/ReadVariableOp2n
5prune_low_magnitude_fc3_prun/LessEqual/ReadVariableOp5prune_low_magnitude_fc3_prun/LessEqual/ReadVariableOp2b
/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp2f
1prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_11prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_12Z
+prune_low_magnitude_fc3_prun/ReadVariableOp+prune_low_magnitude_fc3_prun/ReadVariableOp2^
-prune_low_magnitude_fc3_prun/ReadVariableOp_1-prune_low_magnitude_fc3_prun/ReadVariableOp_12^
-prune_low_magnitude_fc3_prun/ReadVariableOp_2-prune_low_magnitude_fc3_prun/ReadVariableOp_22^
-prune_low_magnitude_fc3_prun/ReadVariableOp_3-prune_low_magnitude_fc3_prun/ReadVariableOp_32^
-prune_low_magnitude_fc3_prun/ReadVariableOp_4-prune_low_magnitude_fc3_prun/ReadVariableOp_42^
-prune_low_magnitude_fc3_prun/ReadVariableOp_5-prune_low_magnitude_fc3_prun/ReadVariableOp_52^
-prune_low_magnitude_fc3_prun/ReadVariableOp_6-prune_low_magnitude_fc3_prun/ReadVariableOp_62b
/prune_low_magnitude_fc3_prun/Sub/ReadVariableOp/prune_low_magnitude_fc3_prun/Sub/ReadVariableOp2�
Dprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuardDprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard2�
@prune_low_magnitude_fc3_prun/assert_greater_equal/ReadVariableOp@prune_low_magnitude_fc3_prun/assert_greater_equal/ReadVariableOp2F
!prune_low_magnitude_fc3_prun/cond!prune_low_magnitude_fc3_prun/cond2l
4prune_low_magnitude_fc4_prunedclass/AssignVariableOp4prune_low_magnitude_fc4_prunedclass/AssignVariableOp2p
6prune_low_magnitude_fc4_prunedclass/AssignVariableOp_16prune_low_magnitude_fc4_prunedclass/AssignVariableOp_12�
?prune_low_magnitude_fc4_prunedclass/GreaterEqual/ReadVariableOp?prune_low_magnitude_fc4_prunedclass/GreaterEqual/ReadVariableOp2|
<prune_low_magnitude_fc4_prunedclass/LessEqual/ReadVariableOp<prune_low_magnitude_fc4_prunedclass/LessEqual/ReadVariableOp2p
6prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp6prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp2t
8prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_18prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_12h
2prune_low_magnitude_fc4_prunedclass/ReadVariableOp2prune_low_magnitude_fc4_prunedclass/ReadVariableOp2l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_14prune_low_magnitude_fc4_prunedclass/ReadVariableOp_12l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_24prune_low_magnitude_fc4_prunedclass/ReadVariableOp_22l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_34prune_low_magnitude_fc4_prunedclass/ReadVariableOp_32l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_44prune_low_magnitude_fc4_prunedclass/ReadVariableOp_42l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_54prune_low_magnitude_fc4_prunedclass/ReadVariableOp_52l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_64prune_low_magnitude_fc4_prunedclass/ReadVariableOp_62p
6prune_low_magnitude_fc4_prunedclass/Sub/ReadVariableOp6prune_low_magnitude_fc4_prunedclass/Sub/ReadVariableOp2�
Kprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuardKprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard2�
Gprune_low_magnitude_fc4_prunedclass/assert_greater_equal/ReadVariableOpGprune_low_magnitude_fc4_prunedclass/assert_greater_equal/ReadVariableOp2T
(prune_low_magnitude_fc4_prunedclass/cond(prune_low_magnitude_fc4_prunedclass/cond2�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Qprune_low_magnitude_fc2_prun_assert_greater_equal_Assert_AssertGuard_true_1597736�
�prune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc2_prun_assert_greater_equal_all
T
Pprune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_placeholder	V
Rprune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_placeholder_1	S
Oprune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_identity_1
�
Iprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Mprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc2_prun_assert_greater_equal_allJ^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Oprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityVprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "�
Oprune_low_magnitude_fc2_prun_assert_greater_equal_assert_assertguard_identity_1Xprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
cond_false_1598938
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
O
	cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 b
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: T
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
�
�
,__inference_classifier_layer_call_fn_1595647
encoder_input
unknown:@
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:(

unknown_13:(

unknown_14:(


unknown_15:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_classifier_layer_call_and_return_conditional_losses_1595610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������@: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�
�
/prune_low_magnitude_fc3_prun_cond_false_15980201
-prune_low_magnitude_fc3_prun_cond_placeholder3
/prune_low_magnitude_fc3_prun_cond_placeholder_13
/prune_low_magnitude_fc3_prun_cond_placeholder_23
/prune_low_magnitude_fc3_prun_cond_placeholder_3X
Tprune_low_magnitude_fc3_prun_cond_identity_prune_low_magnitude_fc3_prun_logicaland_1
0
,prune_low_magnitude_fc3_prun_cond_identity_1
l
&prune_low_magnitude_fc3_prun/cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
*prune_low_magnitude_fc3_prun/cond/IdentityIdentityTprune_low_magnitude_fc3_prun_cond_identity_prune_low_magnitude_fc3_prun_logicaland_1'^prune_low_magnitude_fc3_prun/cond/NoOp*
T0
*
_output_shapes
: �
,prune_low_magnitude_fc3_prun/cond/Identity_1Identity3prune_low_magnitude_fc3_prun/cond/Identity:output:0*
T0
*
_output_shapes
: "e
,prune_low_magnitude_fc3_prun_cond_identity_15prune_low_magnitude_fc3_prun/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
�
�
5assert_greater_equal_Assert_AssertGuard_false_1599252K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
��.assert_greater_equal/Assert/AssertGuard/Assert�
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp/^assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�1
�
@__inference_fc1_layer_call_and_return_conditional_losses_1594900

inputs)
readvariableop_resource:@'
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:@N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:@@
NegNegtruediv:z:0*
T0*
_output_shapes

:@D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:@I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:@N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:@[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:@R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:@P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:@L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:@h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:@M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:@L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:@R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:@h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:@U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
+__inference_fc5_class_layer_call_fn_1599889

inputs
unknown:(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_fc5_class_layer_call_and_return_conditional_losses_1595447o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�>
�
cond_true_15989373
)cond_greaterequal_readvariableop_resource:	 >
,cond_pruning_ops_abs_readvariableop_resource:0
cond_assignvariableop_resource:*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
��cond/AssignVariableOp�cond/AssignVariableOp_1� cond/GreaterEqual/ReadVariableOp�cond/LessEqual/ReadVariableOp�cond/Sub/ReadVariableOp�#cond/pruning_ops/Abs/ReadVariableOp�
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	V
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	S
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N~
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: N
cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�NM
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : ^
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: V
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: `
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: y
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	M

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�f
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: Q
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rdb
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: N
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ^

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: \
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: O

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0q
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:X
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :�m
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: [
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?r
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: q
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: Z
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: _
cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
cond/pruning_ops/MaximumMaximumcond/pruning_ops/Round:y:0#cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: m
cond/pruning_ops/Cast_1Castcond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: q
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:�Z
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :��
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:�:�Z
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: `
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: Z
cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_2Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: b
 cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2_1GatherV2!cond/pruning_ops/TopKV2:indices:0cond/pruning_ops/sub_2:z:0)cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:Z
cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value
B :�`
cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zb
 cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
cond/pruning_ops/one_hotOneHot$cond/pruning_ops/GatherV2_1:output:0 cond/pruning_ops/Size_2:output:0'cond/pruning_ops/one_hot/Const:output:0)cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes	
:�q
 cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
cond/pruning_ops/Reshape_1Reshape!cond/pruning_ops/one_hot:output:0)cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
cond/pruning_ops/LogicalOr	LogicalOr!cond/pruning_ops/GreaterEqual:z:0#cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:i
	cond/CastCastcond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: `
cond/Identity_1Identitycond/Identity:output:0
^cond/NoOp*
T0
*
_output_shapes
: �
	cond/NoOpNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�_
�
G__inference_classifier_layer_call_and_return_conditional_losses_1596788
encoder_input
fc1_1596710:@
fc1_1596712:.
$prune_low_magnitude_fc2_prun_1596716:	 6
$prune_low_magnitude_fc2_prun_1596718:6
$prune_low_magnitude_fc2_prun_1596720:.
$prune_low_magnitude_fc2_prun_1596722: 2
$prune_low_magnitude_fc2_prun_1596724:.
$prune_low_magnitude_fc3_prun_1596728:	 6
$prune_low_magnitude_fc3_prun_1596730:6
$prune_low_magnitude_fc3_prun_1596732:.
$prune_low_magnitude_fc3_prun_1596734: 2
$prune_low_magnitude_fc3_prun_1596736:(
encoder_output_1596740:$
encoder_output_1596742:5
+prune_low_magnitude_fc4_prunedclass_1596745:	 =
+prune_low_magnitude_fc4_prunedclass_1596747:=
+prune_low_magnitude_fc4_prunedclass_1596749:5
+prune_low_magnitude_fc4_prunedclass_1596751: 9
+prune_low_magnitude_fc4_prunedclass_1596753:#
fc5_class_1596757:(
fc5_class_1596759:((
classifier_out_1596763:(
$
classifier_out_1596765:

identity��&classifier_out/StatefulPartitionedCall�4classifier_out/kernel/Regularizer/Abs/ReadVariableOp�&encoder_output/StatefulPartitionedCall�fc1/StatefulPartitionedCall�!fc5_class/StatefulPartitionedCall�/fc5_class/kernel/Regularizer/Abs/ReadVariableOp�4prune_low_magnitude_fc2_prun/StatefulPartitionedCall�4prune_low_magnitude_fc3_prun/StatefulPartitionedCall�;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp�
fc1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputfc1_1596710fc1_1596712*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_1594900�
relu1/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_1594955�
4prune_low_magnitude_fc2_prun/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0$prune_low_magnitude_fc2_prun_1596716$prune_low_magnitude_fc2_prun_1596718$prune_low_magnitude_fc2_prun_1596720$prune_low_magnitude_fc2_prun_1596722$prune_low_magnitude_fc2_prun_1596724*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1596387�
relu2/PartitionedCallPartitionedCall=prune_low_magnitude_fc2_prun/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_1595086�
4prune_low_magnitude_fc3_prun/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0$prune_low_magnitude_fc3_prun_1596728$prune_low_magnitude_fc3_prun_1596730$prune_low_magnitude_fc3_prun_1596732$prune_low_magnitude_fc3_prun_1596734$prune_low_magnitude_fc3_prun_1596736*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1596152�
relu3_enc/PartitionedCallPartitionedCall=prune_low_magnitude_fc3_prun/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1595217�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall"relu3_enc/PartitionedCall:output:0encoder_output_1596740encoder_output_1596742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_output_layer_call_and_return_conditional_losses_1595230�
;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0+prune_low_magnitude_fc4_prunedclass_1596745+prune_low_magnitude_fc4_prunedclass_1596747+prune_low_magnitude_fc4_prunedclass_1596749+prune_low_magnitude_fc4_prunedclass_1596751+prune_low_magnitude_fc4_prunedclass_1596753*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *i
fdRb
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1595907�
prunclass_relu4/PartitionedCallPartitionedCallDprune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1595371�
!fc5_class/StatefulPartitionedCallStatefulPartitionedCall(prunclass_relu4/PartitionedCall:output:0fc5_class_1596757fc5_class_1596759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_fc5_class_layer_call_and_return_conditional_losses_1595447�
class_relu5/PartitionedCallPartitionedCall*fc5_class/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_class_relu5_layer_call_and_return_conditional_losses_1595502�
&classifier_out/StatefulPartitionedCallStatefulPartitionedCall$class_relu5/PartitionedCall:output:0classifier_out_1596763classifier_out_1596765*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_classifier_out_layer_call_and_return_conditional_losses_1595578�
!classifier_output/PartitionedCallPartitionedCall/classifier_out/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_classifier_output_layer_call_and_return_conditional_losses_1595589�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+prune_low_magnitude_fc4_prunedclass_1596747<^prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
/fc5_class/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfc5_class_1596757*
_output_shapes

:(*
dtype0�
 fc5_class/kernel/Regularizer/AbsAbs7fc5_class/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(s
"fc5_class/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 fc5_class/kernel/Regularizer/SumSum$fc5_class/kernel/Regularizer/Abs:y:0+fc5_class/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"fc5_class/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
 fc5_class/kernel/Regularizer/mulMul+fc5_class/kernel/Regularizer/mul/x:output:0)fc5_class/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4classifier_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpclassifier_out_1596763*
_output_shapes

:(
*
dtype0�
%classifier_out/kernel/Regularizer/AbsAbs<classifier_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(
x
'classifier_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%classifier_out/kernel/Regularizer/SumSum)classifier_out/kernel/Regularizer/Abs:y:00classifier_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'classifier_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%classifier_out/kernel/Regularizer/mulMul0classifier_out/kernel/Regularizer/mul/x:output:0.classifier_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*classifier_output/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp'^classifier_out/StatefulPartitionedCall5^classifier_out/kernel/Regularizer/Abs/ReadVariableOp'^encoder_output/StatefulPartitionedCall^fc1/StatefulPartitionedCall"^fc5_class/StatefulPartitionedCall0^fc5_class/kernel/Regularizer/Abs/ReadVariableOp5^prune_low_magnitude_fc2_prun/StatefulPartitionedCall5^prune_low_magnitude_fc3_prun/StatefulPartitionedCall<^prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCallJ^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������@: : : : : : : : : : : : : : : : : : : : : : : 2P
&classifier_out/StatefulPartitionedCall&classifier_out/StatefulPartitionedCall2l
4classifier_out/kernel/Regularizer/Abs/ReadVariableOp4classifier_out/kernel/Regularizer/Abs/ReadVariableOp2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2F
!fc5_class/StatefulPartitionedCall!fc5_class/StatefulPartitionedCall2b
/fc5_class/kernel/Regularizer/Abs/ReadVariableOp/fc5_class/kernel/Regularizer/Abs/ReadVariableOp2l
4prune_low_magnitude_fc2_prun/StatefulPartitionedCall4prune_low_magnitude_fc2_prun/StatefulPartitionedCall2l
4prune_low_magnitude_fc3_prun/StatefulPartitionedCall4prune_low_magnitude_fc3_prun/StatefulPartitionedCall2z
;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall2�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�
�
5assert_greater_equal_Assert_AssertGuard_false_1598898K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
��.assert_greater_equal/Assert/AssertGuard/Assert�
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp/^assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
cond_false_1599672
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
O
	cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 b
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: T
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
��
�
G__inference_classifier_layer_call_and_return_conditional_losses_1597608

inputs-
fc1_readvariableop_resource:@+
fc1_readvariableop_3_resource:J
8prune_low_magnitude_fc2_prun_mul_readvariableop_resource:L
:prune_low_magnitude_fc2_prun_mul_readvariableop_1_resource:D
6prune_low_magnitude_fc2_prun_readvariableop_3_resource:J
8prune_low_magnitude_fc3_prun_mul_readvariableop_resource:L
:prune_low_magnitude_fc3_prun_mul_readvariableop_1_resource:D
6prune_low_magnitude_fc3_prun_readvariableop_3_resource:?
-encoder_output_matmul_readvariableop_resource:<
.encoder_output_biasadd_readvariableop_resource:Q
?prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource:S
Aprune_low_magnitude_fc4_prunedclass_mul_readvariableop_1_resource:K
=prune_low_magnitude_fc4_prunedclass_readvariableop_3_resource:3
!fc5_class_readvariableop_resource:(1
#fc5_class_readvariableop_3_resource:(8
&classifier_out_readvariableop_resource:(
6
(classifier_out_readvariableop_3_resource:

identity��classifier_out/ReadVariableOp�classifier_out/ReadVariableOp_1�classifier_out/ReadVariableOp_2�classifier_out/ReadVariableOp_3�classifier_out/ReadVariableOp_4�classifier_out/ReadVariableOp_5�4classifier_out/kernel/Regularizer/Abs/ReadVariableOp�%encoder_output/BiasAdd/ReadVariableOp�$encoder_output/MatMul/ReadVariableOp�fc1/ReadVariableOp�fc1/ReadVariableOp_1�fc1/ReadVariableOp_2�fc1/ReadVariableOp_3�fc1/ReadVariableOp_4�fc1/ReadVariableOp_5�fc5_class/ReadVariableOp�fc5_class/ReadVariableOp_1�fc5_class/ReadVariableOp_2�fc5_class/ReadVariableOp_3�fc5_class/ReadVariableOp_4�fc5_class/ReadVariableOp_5�/fc5_class/kernel/Regularizer/Abs/ReadVariableOp�-prune_low_magnitude_fc2_prun/AssignVariableOp�/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp�1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1�+prune_low_magnitude_fc2_prun/ReadVariableOp�-prune_low_magnitude_fc2_prun/ReadVariableOp_1�-prune_low_magnitude_fc2_prun/ReadVariableOp_2�-prune_low_magnitude_fc2_prun/ReadVariableOp_3�-prune_low_magnitude_fc2_prun/ReadVariableOp_4�-prune_low_magnitude_fc2_prun/ReadVariableOp_5�-prune_low_magnitude_fc3_prun/AssignVariableOp�/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp�1prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1�+prune_low_magnitude_fc3_prun/ReadVariableOp�-prune_low_magnitude_fc3_prun/ReadVariableOp_1�-prune_low_magnitude_fc3_prun/ReadVariableOp_2�-prune_low_magnitude_fc3_prun/ReadVariableOp_3�-prune_low_magnitude_fc3_prun/ReadVariableOp_4�-prune_low_magnitude_fc3_prun/ReadVariableOp_5�4prune_low_magnitude_fc4_prunedclass/AssignVariableOp�6prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp�8prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1�2prune_low_magnitude_fc4_prunedclass/ReadVariableOp�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4�4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpK
	fc1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :K
	fc1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : W
fc1/PowPowfc1/Pow/x:output:0fc1/Pow/y:output:0*
T0*
_output_shapes
: M
fc1/CastCastfc1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: n
fc1/ReadVariableOpReadVariableOpfc1_readvariableop_resource*
_output_shapes

:@*
dtype0N
	fc1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ag
fc1/mulMulfc1/ReadVariableOp:value:0fc1/mul/y:output:0*
T0*
_output_shapes

:@Z
fc1/truedivRealDivfc1/mul:z:0fc1/Cast:y:0*
T0*
_output_shapes

:@H
fc1/NegNegfc1/truediv:z:0*
T0*
_output_shapes

:@L
	fc1/RoundRoundfc1/truediv:z:0*
T0*
_output_shapes

:@U
fc1/addAddV2fc1/Neg:y:0fc1/Round:y:0*
T0*
_output_shapes

:@V
fc1/StopGradientStopGradientfc1/add:z:0*
T0*
_output_shapes

:@g
	fc1/add_1AddV2fc1/truediv:z:0fc1/StopGradient:output:0*
T0*
_output_shapes

:@`
fc1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
fc1/clip_by_value/MinimumMinimumfc1/add_1:z:0$fc1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@X
fc1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
fc1/clip_by_valueMaximumfc1/clip_by_value/Minimum:z:0fc1/clip_by_value/y:output:0*
T0*
_output_shapes

:@^
	fc1/mul_1Mulfc1/Cast:y:0fc1/clip_by_value:z:0*
T0*
_output_shapes

:@T
fc1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aj
fc1/truediv_1RealDivfc1/mul_1:z:0fc1/truediv_1/y:output:0*
T0*
_output_shapes

:@P
fc1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	fc1/mul_2Mulfc1/mul_2/x:output:0fc1/truediv_1:z:0*
T0*
_output_shapes

:@p
fc1/ReadVariableOp_1ReadVariableOpfc1_readvariableop_resource*
_output_shapes

:@*
dtype0W
	fc1/Neg_1Negfc1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:@Y
	fc1/add_2AddV2fc1/Neg_1:y:0fc1/mul_2:z:0*
T0*
_output_shapes

:@P
fc1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
	fc1/mul_3Mulfc1/mul_3/x:output:0fc1/add_2:z:0*
T0*
_output_shapes

:@Z
fc1/StopGradient_1StopGradientfc1/mul_3:z:0*
T0*
_output_shapes

:@p
fc1/ReadVariableOp_2ReadVariableOpfc1_readvariableop_resource*
_output_shapes

:@*
dtype0v
	fc1/add_3AddV2fc1/ReadVariableOp_2:value:0fc1/StopGradient_1:output:0*
T0*
_output_shapes

:@]

fc1/MatMulMatMulinputsfc1/add_3:z:0*
T0*'
_output_shapes
:���������M
fc1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :M
fc1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
	fc1/Pow_1Powfc1/Pow_1/x:output:0fc1/Pow_1/y:output:0*
T0*
_output_shapes
: Q

fc1/Cast_1Castfc1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: n
fc1/ReadVariableOp_3ReadVariableOpfc1_readvariableop_3_resource*
_output_shapes
:*
dtype0P
fc1/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ai
	fc1/mul_4Mulfc1/ReadVariableOp_3:value:0fc1/mul_4/y:output:0*
T0*
_output_shapes
:\
fc1/truediv_2RealDivfc1/mul_4:z:0fc1/Cast_1:y:0*
T0*
_output_shapes
:H
	fc1/Neg_2Negfc1/truediv_2:z:0*
T0*
_output_shapes
:L
fc1/Round_1Roundfc1/truediv_2:z:0*
T0*
_output_shapes
:W
	fc1/add_4AddV2fc1/Neg_2:y:0fc1/Round_1:y:0*
T0*
_output_shapes
:V
fc1/StopGradient_2StopGradientfc1/add_4:z:0*
T0*
_output_shapes
:g
	fc1/add_5AddV2fc1/truediv_2:z:0fc1/StopGradient_2:output:0*
T0*
_output_shapes
:b
fc1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
fc1/clip_by_value_1/MinimumMinimumfc1/add_5:z:0&fc1/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:Z
fc1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
fc1/clip_by_value_1Maximumfc1/clip_by_value_1/Minimum:z:0fc1/clip_by_value_1/y:output:0*
T0*
_output_shapes
:^
	fc1/mul_5Mulfc1/Cast_1:y:0fc1/clip_by_value_1:z:0*
T0*
_output_shapes
:T
fc1/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Af
fc1/truediv_3RealDivfc1/mul_5:z:0fc1/truediv_3/y:output:0*
T0*
_output_shapes
:P
fc1/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
	fc1/mul_6Mulfc1/mul_6/x:output:0fc1/truediv_3:z:0*
T0*
_output_shapes
:n
fc1/ReadVariableOp_4ReadVariableOpfc1_readvariableop_3_resource*
_output_shapes
:*
dtype0S
	fc1/Neg_3Negfc1/ReadVariableOp_4:value:0*
T0*
_output_shapes
:U
	fc1/add_6AddV2fc1/Neg_3:y:0fc1/mul_6:z:0*
T0*
_output_shapes
:P
fc1/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
	fc1/mul_7Mulfc1/mul_7/x:output:0fc1/add_6:z:0*
T0*
_output_shapes
:V
fc1/StopGradient_3StopGradientfc1/mul_7:z:0*
T0*
_output_shapes
:n
fc1/ReadVariableOp_5ReadVariableOpfc1_readvariableop_3_resource*
_output_shapes
:*
dtype0r
	fc1/add_7AddV2fc1/ReadVariableOp_5:value:0fc1/StopGradient_3:output:0*
T0*
_output_shapes
:m
fc1/BiasAddBiasAddfc1/MatMul:product:0fc1/add_7:z:0*
T0*'
_output_shapes
:���������M
relu1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :M
relu1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :]
	relu1/PowPowrelu1/Pow/x:output:0relu1/Pow/y:output:0*
T0*
_output_shapes
: Q

relu1/CastCastrelu1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: O
relu1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :O
relu1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : c
relu1/Pow_1Powrelu1/Pow_1/x:output:0relu1/Pow_1/y:output:0*
T0*
_output_shapes
: U
relu1/Cast_1Castrelu1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: P
relu1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @P
relu1/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : ]
relu1/Cast_2Castrelu1/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: P
relu1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Y
	relu1/subSubrelu1/Cast_2:y:0relu1/sub/y:output:0*
T0*
_output_shapes
: X
relu1/Pow_2Powrelu1/Const:output:0relu1/sub:z:0*
T0*
_output_shapes
: V
relu1/sub_1Subrelu1/Cast_1:y:0relu1/Pow_2:z:0*
T0*
_output_shapes
: u
relu1/LessEqual	LessEqualfc1/BiasAdd:output:0relu1/sub_1:z:0*
T0*'
_output_shapes
:���������Z

relu1/ReluRelufc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������Y
relu1/ones_like/ShapeShapefc1/BiasAdd:output:0*
T0*
_output_shapes
:Z
relu1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
relu1/ones_likeFillrelu1/ones_like/Shape:output:0relu1/ones_like/Const:output:0*
T0*'
_output_shapes
:���������V
relu1/sub_2Subrelu1/Cast_1:y:0relu1/Pow_2:z:0*
T0*
_output_shapes
: m
	relu1/mulMulrelu1/ones_like:output:0relu1/sub_2:z:0*
T0*'
_output_shapes
:����������
relu1/SelectV2SelectV2relu1/LessEqual:z:0relu1/Relu:activations:0relu1/mul:z:0*
T0*'
_output_shapes
:���������j
relu1/mul_1Mulfc1/BiasAdd:output:0relu1/Cast:y:0*
T0*'
_output_shapes
:���������m
relu1/truedivRealDivrelu1/mul_1:z:0relu1/Cast_1:y:0*
T0*'
_output_shapes
:���������U
	relu1/NegNegrelu1/truediv:z:0*
T0*'
_output_shapes
:���������Y
relu1/RoundRoundrelu1/truediv:z:0*
T0*'
_output_shapes
:���������d
	relu1/addAddV2relu1/Neg:y:0relu1/Round:y:0*
T0*'
_output_shapes
:���������c
relu1/StopGradientStopGradientrelu1/add:z:0*
T0*'
_output_shapes
:���������v
relu1/add_1AddV2relu1/truediv:z:0relu1/StopGradient:output:0*
T0*'
_output_shapes
:���������m
relu1/truediv_1RealDivrelu1/add_1:z:0relu1/Cast:y:0*
T0*'
_output_shapes
:���������V
relu1/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
relu1/truediv_2RealDivrelu1/truediv_2/x:output:0relu1/Cast:y:0*
T0*
_output_shapes
: R
relu1/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?`
relu1/sub_3Subrelu1/sub_3/x:output:0relu1/truediv_2:z:0*
T0*
_output_shapes
: ~
relu1/clip_by_value/MinimumMinimumrelu1/truediv_1:z:0relu1/sub_3:z:0*
T0*'
_output_shapes
:���������Z
relu1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
relu1/clip_by_valueMaximumrelu1/clip_by_value/Minimum:z:0relu1/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������o
relu1/mul_2Mulrelu1/Cast_1:y:0relu1/clip_by_value:z:0*
T0*'
_output_shapes
:���������]
relu1/Neg_1Negrelu1/SelectV2:output:0*
T0*'
_output_shapes
:���������h
relu1/add_2AddV2relu1/Neg_1:y:0relu1/mul_2:z:0*
T0*'
_output_shapes
:���������R
relu1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?m
relu1/mul_3Mulrelu1/mul_3/x:output:0relu1/add_2:z:0*
T0*'
_output_shapes
:���������g
relu1/StopGradient_1StopGradientrelu1/mul_3:z:0*
T0*'
_output_shapes
:���������~
relu1/add_3AddV2relu1/SelectV2:output:0relu1/StopGradient_1:output:0*
T0*'
_output_shapes
:���������D
&prune_low_magnitude_fc2_prun/no_updateNoOp*
_output_shapes
 F
(prune_low_magnitude_fc2_prun/no_update_1NoOp*
_output_shapes
 �
/prune_low_magnitude_fc2_prun/Mul/ReadVariableOpReadVariableOp8prune_low_magnitude_fc2_prun_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1ReadVariableOp:prune_low_magnitude_fc2_prun_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
 prune_low_magnitude_fc2_prun/MulMul7prune_low_magnitude_fc2_prun/Mul/ReadVariableOp:value:09prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc2_prun/AssignVariableOpAssignVariableOp8prune_low_magnitude_fc2_prun_mul_readvariableop_resource$prune_low_magnitude_fc2_prun/Mul:z:00^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
'prune_low_magnitude_fc2_prun/group_depsNoOp.^prune_low_magnitude_fc2_prun/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
)prune_low_magnitude_fc2_prun/group_deps_1NoOp(^prune_low_magnitude_fc2_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 d
"prune_low_magnitude_fc2_prun/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :d
"prune_low_magnitude_fc2_prun/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
 prune_low_magnitude_fc2_prun/PowPow+prune_low_magnitude_fc2_prun/Pow/x:output:0+prune_low_magnitude_fc2_prun/Pow/y:output:0*
T0*
_output_shapes
: 
!prune_low_magnitude_fc2_prun/CastCast$prune_low_magnitude_fc2_prun/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
+prune_low_magnitude_fc2_prun/ReadVariableOpReadVariableOp8prune_low_magnitude_fc2_prun_mul_readvariableop_resource.^prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes

:*
dtype0i
$prune_low_magnitude_fc2_prun/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
"prune_low_magnitude_fc2_prun/mul_1Mul3prune_low_magnitude_fc2_prun/ReadVariableOp:value:0-prune_low_magnitude_fc2_prun/mul_1/y:output:0*
T0*
_output_shapes

:�
$prune_low_magnitude_fc2_prun/truedivRealDiv&prune_low_magnitude_fc2_prun/mul_1:z:0%prune_low_magnitude_fc2_prun/Cast:y:0*
T0*
_output_shapes

:z
 prune_low_magnitude_fc2_prun/NegNeg(prune_low_magnitude_fc2_prun/truediv:z:0*
T0*
_output_shapes

:~
"prune_low_magnitude_fc2_prun/RoundRound(prune_low_magnitude_fc2_prun/truediv:z:0*
T0*
_output_shapes

:�
 prune_low_magnitude_fc2_prun/addAddV2$prune_low_magnitude_fc2_prun/Neg:y:0&prune_low_magnitude_fc2_prun/Round:y:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc2_prun/StopGradientStopGradient$prune_low_magnitude_fc2_prun/add:z:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc2_prun/add_1AddV2(prune_low_magnitude_fc2_prun/truediv:z:02prune_low_magnitude_fc2_prun/StopGradient:output:0*
T0*
_output_shapes

:y
4prune_low_magnitude_fc2_prun/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
2prune_low_magnitude_fc2_prun/clip_by_value/MinimumMinimum&prune_low_magnitude_fc2_prun/add_1:z:0=prune_low_magnitude_fc2_prun/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:q
,prune_low_magnitude_fc2_prun/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
*prune_low_magnitude_fc2_prun/clip_by_valueMaximum6prune_low_magnitude_fc2_prun/clip_by_value/Minimum:z:05prune_low_magnitude_fc2_prun/clip_by_value/y:output:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc2_prun/mul_2Mul%prune_low_magnitude_fc2_prun/Cast:y:0.prune_low_magnitude_fc2_prun/clip_by_value:z:0*
T0*
_output_shapes

:m
(prune_low_magnitude_fc2_prun/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
&prune_low_magnitude_fc2_prun/truediv_1RealDiv&prune_low_magnitude_fc2_prun/mul_2:z:01prune_low_magnitude_fc2_prun/truediv_1/y:output:0*
T0*
_output_shapes

:i
$prune_low_magnitude_fc2_prun/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc2_prun/mul_3Mul-prune_low_magnitude_fc2_prun/mul_3/x:output:0*prune_low_magnitude_fc2_prun/truediv_1:z:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc2_prun/ReadVariableOp_1ReadVariableOp8prune_low_magnitude_fc2_prun_mul_readvariableop_resource.^prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc2_prun/Neg_1Neg5prune_low_magnitude_fc2_prun/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc2_prun/add_2AddV2&prune_low_magnitude_fc2_prun/Neg_1:y:0&prune_low_magnitude_fc2_prun/mul_3:z:0*
T0*
_output_shapes

:i
$prune_low_magnitude_fc2_prun/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc2_prun/mul_4Mul-prune_low_magnitude_fc2_prun/mul_4/x:output:0&prune_low_magnitude_fc2_prun/add_2:z:0*
T0*
_output_shapes

:�
+prune_low_magnitude_fc2_prun/StopGradient_1StopGradient&prune_low_magnitude_fc2_prun/mul_4:z:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc2_prun/ReadVariableOp_2ReadVariableOp8prune_low_magnitude_fc2_prun_mul_readvariableop_resource.^prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc2_prun/add_3AddV25prune_low_magnitude_fc2_prun/ReadVariableOp_2:value:04prune_low_magnitude_fc2_prun/StopGradient_1:output:0*
T0*
_output_shapes

:�
#prune_low_magnitude_fc2_prun/MatMulMatMulrelu1/add_3:z:0&prune_low_magnitude_fc2_prun/add_3:z:0*
T0*'
_output_shapes
:���������f
$prune_low_magnitude_fc2_prun/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :f
$prune_low_magnitude_fc2_prun/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
"prune_low_magnitude_fc2_prun/Pow_1Pow-prune_low_magnitude_fc2_prun/Pow_1/x:output:0-prune_low_magnitude_fc2_prun/Pow_1/y:output:0*
T0*
_output_shapes
: �
#prune_low_magnitude_fc2_prun/Cast_1Cast&prune_low_magnitude_fc2_prun/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
-prune_low_magnitude_fc2_prun/ReadVariableOp_3ReadVariableOp6prune_low_magnitude_fc2_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0i
$prune_low_magnitude_fc2_prun/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
"prune_low_magnitude_fc2_prun/mul_5Mul5prune_low_magnitude_fc2_prun/ReadVariableOp_3:value:0-prune_low_magnitude_fc2_prun/mul_5/y:output:0*
T0*
_output_shapes
:�
&prune_low_magnitude_fc2_prun/truediv_2RealDiv&prune_low_magnitude_fc2_prun/mul_5:z:0'prune_low_magnitude_fc2_prun/Cast_1:y:0*
T0*
_output_shapes
:z
"prune_low_magnitude_fc2_prun/Neg_2Neg*prune_low_magnitude_fc2_prun/truediv_2:z:0*
T0*
_output_shapes
:~
$prune_low_magnitude_fc2_prun/Round_1Round*prune_low_magnitude_fc2_prun/truediv_2:z:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc2_prun/add_4AddV2&prune_low_magnitude_fc2_prun/Neg_2:y:0(prune_low_magnitude_fc2_prun/Round_1:y:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc2_prun/StopGradient_2StopGradient&prune_low_magnitude_fc2_prun/add_4:z:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc2_prun/add_5AddV2*prune_low_magnitude_fc2_prun/truediv_2:z:04prune_low_magnitude_fc2_prun/StopGradient_2:output:0*
T0*
_output_shapes
:{
6prune_low_magnitude_fc2_prun/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
4prune_low_magnitude_fc2_prun/clip_by_value_1/MinimumMinimum&prune_low_magnitude_fc2_prun/add_5:z:0?prune_low_magnitude_fc2_prun/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:s
.prune_low_magnitude_fc2_prun/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
,prune_low_magnitude_fc2_prun/clip_by_value_1Maximum8prune_low_magnitude_fc2_prun/clip_by_value_1/Minimum:z:07prune_low_magnitude_fc2_prun/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc2_prun/mul_6Mul'prune_low_magnitude_fc2_prun/Cast_1:y:00prune_low_magnitude_fc2_prun/clip_by_value_1:z:0*
T0*
_output_shapes
:m
(prune_low_magnitude_fc2_prun/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
&prune_low_magnitude_fc2_prun/truediv_3RealDiv&prune_low_magnitude_fc2_prun/mul_6:z:01prune_low_magnitude_fc2_prun/truediv_3/y:output:0*
T0*
_output_shapes
:i
$prune_low_magnitude_fc2_prun/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc2_prun/mul_7Mul-prune_low_magnitude_fc2_prun/mul_7/x:output:0*prune_low_magnitude_fc2_prun/truediv_3:z:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc2_prun/ReadVariableOp_4ReadVariableOp6prune_low_magnitude_fc2_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0�
"prune_low_magnitude_fc2_prun/Neg_3Neg5prune_low_magnitude_fc2_prun/ReadVariableOp_4:value:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc2_prun/add_6AddV2&prune_low_magnitude_fc2_prun/Neg_3:y:0&prune_low_magnitude_fc2_prun/mul_7:z:0*
T0*
_output_shapes
:i
$prune_low_magnitude_fc2_prun/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc2_prun/mul_8Mul-prune_low_magnitude_fc2_prun/mul_8/x:output:0&prune_low_magnitude_fc2_prun/add_6:z:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc2_prun/StopGradient_3StopGradient&prune_low_magnitude_fc2_prun/mul_8:z:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc2_prun/ReadVariableOp_5ReadVariableOp6prune_low_magnitude_fc2_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0�
"prune_low_magnitude_fc2_prun/add_7AddV25prune_low_magnitude_fc2_prun/ReadVariableOp_5:value:04prune_low_magnitude_fc2_prun/StopGradient_3:output:0*
T0*
_output_shapes
:�
$prune_low_magnitude_fc2_prun/BiasAddBiasAdd-prune_low_magnitude_fc2_prun/MatMul:product:0&prune_low_magnitude_fc2_prun/add_7:z:0*
T0*'
_output_shapes
:���������M
relu2/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :M
relu2/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :]
	relu2/PowPowrelu2/Pow/x:output:0relu2/Pow/y:output:0*
T0*
_output_shapes
: Q

relu2/CastCastrelu2/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: O
relu2/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :O
relu2/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : c
relu2/Pow_1Powrelu2/Pow_1/x:output:0relu2/Pow_1/y:output:0*
T0*
_output_shapes
: U
relu2/Cast_1Castrelu2/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: P
relu2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @P
relu2/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : ]
relu2/Cast_2Castrelu2/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: P
relu2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Y
	relu2/subSubrelu2/Cast_2:y:0relu2/sub/y:output:0*
T0*
_output_shapes
: X
relu2/Pow_2Powrelu2/Const:output:0relu2/sub:z:0*
T0*
_output_shapes
: V
relu2/sub_1Subrelu2/Cast_1:y:0relu2/Pow_2:z:0*
T0*
_output_shapes
: �
relu2/LessEqual	LessEqual-prune_low_magnitude_fc2_prun/BiasAdd:output:0relu2/sub_1:z:0*
T0*'
_output_shapes
:���������s

relu2/ReluRelu-prune_low_magnitude_fc2_prun/BiasAdd:output:0*
T0*'
_output_shapes
:���������r
relu2/ones_like/ShapeShape-prune_low_magnitude_fc2_prun/BiasAdd:output:0*
T0*
_output_shapes
:Z
relu2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
relu2/ones_likeFillrelu2/ones_like/Shape:output:0relu2/ones_like/Const:output:0*
T0*'
_output_shapes
:���������V
relu2/sub_2Subrelu2/Cast_1:y:0relu2/Pow_2:z:0*
T0*
_output_shapes
: m
	relu2/mulMulrelu2/ones_like:output:0relu2/sub_2:z:0*
T0*'
_output_shapes
:����������
relu2/SelectV2SelectV2relu2/LessEqual:z:0relu2/Relu:activations:0relu2/mul:z:0*
T0*'
_output_shapes
:����������
relu2/mul_1Mul-prune_low_magnitude_fc2_prun/BiasAdd:output:0relu2/Cast:y:0*
T0*'
_output_shapes
:���������m
relu2/truedivRealDivrelu2/mul_1:z:0relu2/Cast_1:y:0*
T0*'
_output_shapes
:���������U
	relu2/NegNegrelu2/truediv:z:0*
T0*'
_output_shapes
:���������Y
relu2/RoundRoundrelu2/truediv:z:0*
T0*'
_output_shapes
:���������d
	relu2/addAddV2relu2/Neg:y:0relu2/Round:y:0*
T0*'
_output_shapes
:���������c
relu2/StopGradientStopGradientrelu2/add:z:0*
T0*'
_output_shapes
:���������v
relu2/add_1AddV2relu2/truediv:z:0relu2/StopGradient:output:0*
T0*'
_output_shapes
:���������m
relu2/truediv_1RealDivrelu2/add_1:z:0relu2/Cast:y:0*
T0*'
_output_shapes
:���������V
relu2/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
relu2/truediv_2RealDivrelu2/truediv_2/x:output:0relu2/Cast:y:0*
T0*
_output_shapes
: R
relu2/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?`
relu2/sub_3Subrelu2/sub_3/x:output:0relu2/truediv_2:z:0*
T0*
_output_shapes
: ~
relu2/clip_by_value/MinimumMinimumrelu2/truediv_1:z:0relu2/sub_3:z:0*
T0*'
_output_shapes
:���������Z
relu2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
relu2/clip_by_valueMaximumrelu2/clip_by_value/Minimum:z:0relu2/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������o
relu2/mul_2Mulrelu2/Cast_1:y:0relu2/clip_by_value:z:0*
T0*'
_output_shapes
:���������]
relu2/Neg_1Negrelu2/SelectV2:output:0*
T0*'
_output_shapes
:���������h
relu2/add_2AddV2relu2/Neg_1:y:0relu2/mul_2:z:0*
T0*'
_output_shapes
:���������R
relu2/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?m
relu2/mul_3Mulrelu2/mul_3/x:output:0relu2/add_2:z:0*
T0*'
_output_shapes
:���������g
relu2/StopGradient_1StopGradientrelu2/mul_3:z:0*
T0*'
_output_shapes
:���������~
relu2/add_3AddV2relu2/SelectV2:output:0relu2/StopGradient_1:output:0*
T0*'
_output_shapes
:���������D
&prune_low_magnitude_fc3_prun/no_updateNoOp*
_output_shapes
 F
(prune_low_magnitude_fc3_prun/no_update_1NoOp*
_output_shapes
 �
/prune_low_magnitude_fc3_prun/Mul/ReadVariableOpReadVariableOp8prune_low_magnitude_fc3_prun_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
1prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1ReadVariableOp:prune_low_magnitude_fc3_prun_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
 prune_low_magnitude_fc3_prun/MulMul7prune_low_magnitude_fc3_prun/Mul/ReadVariableOp:value:09prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc3_prun/AssignVariableOpAssignVariableOp8prune_low_magnitude_fc3_prun_mul_readvariableop_resource$prune_low_magnitude_fc3_prun/Mul:z:00^prune_low_magnitude_fc3_prun/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
'prune_low_magnitude_fc3_prun/group_depsNoOp.^prune_low_magnitude_fc3_prun/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
)prune_low_magnitude_fc3_prun/group_deps_1NoOp(^prune_low_magnitude_fc3_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 d
"prune_low_magnitude_fc3_prun/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :d
"prune_low_magnitude_fc3_prun/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
 prune_low_magnitude_fc3_prun/PowPow+prune_low_magnitude_fc3_prun/Pow/x:output:0+prune_low_magnitude_fc3_prun/Pow/y:output:0*
T0*
_output_shapes
: 
!prune_low_magnitude_fc3_prun/CastCast$prune_low_magnitude_fc3_prun/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
+prune_low_magnitude_fc3_prun/ReadVariableOpReadVariableOp8prune_low_magnitude_fc3_prun_mul_readvariableop_resource.^prune_low_magnitude_fc3_prun/AssignVariableOp*
_output_shapes

:*
dtype0i
$prune_low_magnitude_fc3_prun/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
"prune_low_magnitude_fc3_prun/mul_1Mul3prune_low_magnitude_fc3_prun/ReadVariableOp:value:0-prune_low_magnitude_fc3_prun/mul_1/y:output:0*
T0*
_output_shapes

:�
$prune_low_magnitude_fc3_prun/truedivRealDiv&prune_low_magnitude_fc3_prun/mul_1:z:0%prune_low_magnitude_fc3_prun/Cast:y:0*
T0*
_output_shapes

:z
 prune_low_magnitude_fc3_prun/NegNeg(prune_low_magnitude_fc3_prun/truediv:z:0*
T0*
_output_shapes

:~
"prune_low_magnitude_fc3_prun/RoundRound(prune_low_magnitude_fc3_prun/truediv:z:0*
T0*
_output_shapes

:�
 prune_low_magnitude_fc3_prun/addAddV2$prune_low_magnitude_fc3_prun/Neg:y:0&prune_low_magnitude_fc3_prun/Round:y:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc3_prun/StopGradientStopGradient$prune_low_magnitude_fc3_prun/add:z:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc3_prun/add_1AddV2(prune_low_magnitude_fc3_prun/truediv:z:02prune_low_magnitude_fc3_prun/StopGradient:output:0*
T0*
_output_shapes

:y
4prune_low_magnitude_fc3_prun/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
2prune_low_magnitude_fc3_prun/clip_by_value/MinimumMinimum&prune_low_magnitude_fc3_prun/add_1:z:0=prune_low_magnitude_fc3_prun/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:q
,prune_low_magnitude_fc3_prun/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
*prune_low_magnitude_fc3_prun/clip_by_valueMaximum6prune_low_magnitude_fc3_prun/clip_by_value/Minimum:z:05prune_low_magnitude_fc3_prun/clip_by_value/y:output:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc3_prun/mul_2Mul%prune_low_magnitude_fc3_prun/Cast:y:0.prune_low_magnitude_fc3_prun/clip_by_value:z:0*
T0*
_output_shapes

:m
(prune_low_magnitude_fc3_prun/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
&prune_low_magnitude_fc3_prun/truediv_1RealDiv&prune_low_magnitude_fc3_prun/mul_2:z:01prune_low_magnitude_fc3_prun/truediv_1/y:output:0*
T0*
_output_shapes

:i
$prune_low_magnitude_fc3_prun/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc3_prun/mul_3Mul-prune_low_magnitude_fc3_prun/mul_3/x:output:0*prune_low_magnitude_fc3_prun/truediv_1:z:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc3_prun/ReadVariableOp_1ReadVariableOp8prune_low_magnitude_fc3_prun_mul_readvariableop_resource.^prune_low_magnitude_fc3_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc3_prun/Neg_1Neg5prune_low_magnitude_fc3_prun/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
"prune_low_magnitude_fc3_prun/add_2AddV2&prune_low_magnitude_fc3_prun/Neg_1:y:0&prune_low_magnitude_fc3_prun/mul_3:z:0*
T0*
_output_shapes

:i
$prune_low_magnitude_fc3_prun/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc3_prun/mul_4Mul-prune_low_magnitude_fc3_prun/mul_4/x:output:0&prune_low_magnitude_fc3_prun/add_2:z:0*
T0*
_output_shapes

:�
+prune_low_magnitude_fc3_prun/StopGradient_1StopGradient&prune_low_magnitude_fc3_prun/mul_4:z:0*
T0*
_output_shapes

:�
-prune_low_magnitude_fc3_prun/ReadVariableOp_2ReadVariableOp8prune_low_magnitude_fc3_prun_mul_readvariableop_resource.^prune_low_magnitude_fc3_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc3_prun/add_3AddV25prune_low_magnitude_fc3_prun/ReadVariableOp_2:value:04prune_low_magnitude_fc3_prun/StopGradient_1:output:0*
T0*
_output_shapes

:�
#prune_low_magnitude_fc3_prun/MatMulMatMulrelu2/add_3:z:0&prune_low_magnitude_fc3_prun/add_3:z:0*
T0*'
_output_shapes
:���������f
$prune_low_magnitude_fc3_prun/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :f
$prune_low_magnitude_fc3_prun/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
"prune_low_magnitude_fc3_prun/Pow_1Pow-prune_low_magnitude_fc3_prun/Pow_1/x:output:0-prune_low_magnitude_fc3_prun/Pow_1/y:output:0*
T0*
_output_shapes
: �
#prune_low_magnitude_fc3_prun/Cast_1Cast&prune_low_magnitude_fc3_prun/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
-prune_low_magnitude_fc3_prun/ReadVariableOp_3ReadVariableOp6prune_low_magnitude_fc3_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0i
$prune_low_magnitude_fc3_prun/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
"prune_low_magnitude_fc3_prun/mul_5Mul5prune_low_magnitude_fc3_prun/ReadVariableOp_3:value:0-prune_low_magnitude_fc3_prun/mul_5/y:output:0*
T0*
_output_shapes
:�
&prune_low_magnitude_fc3_prun/truediv_2RealDiv&prune_low_magnitude_fc3_prun/mul_5:z:0'prune_low_magnitude_fc3_prun/Cast_1:y:0*
T0*
_output_shapes
:z
"prune_low_magnitude_fc3_prun/Neg_2Neg*prune_low_magnitude_fc3_prun/truediv_2:z:0*
T0*
_output_shapes
:~
$prune_low_magnitude_fc3_prun/Round_1Round*prune_low_magnitude_fc3_prun/truediv_2:z:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc3_prun/add_4AddV2&prune_low_magnitude_fc3_prun/Neg_2:y:0(prune_low_magnitude_fc3_prun/Round_1:y:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc3_prun/StopGradient_2StopGradient&prune_low_magnitude_fc3_prun/add_4:z:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc3_prun/add_5AddV2*prune_low_magnitude_fc3_prun/truediv_2:z:04prune_low_magnitude_fc3_prun/StopGradient_2:output:0*
T0*
_output_shapes
:{
6prune_low_magnitude_fc3_prun/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
4prune_low_magnitude_fc3_prun/clip_by_value_1/MinimumMinimum&prune_low_magnitude_fc3_prun/add_5:z:0?prune_low_magnitude_fc3_prun/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:s
.prune_low_magnitude_fc3_prun/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
,prune_low_magnitude_fc3_prun/clip_by_value_1Maximum8prune_low_magnitude_fc3_prun/clip_by_value_1/Minimum:z:07prune_low_magnitude_fc3_prun/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc3_prun/mul_6Mul'prune_low_magnitude_fc3_prun/Cast_1:y:00prune_low_magnitude_fc3_prun/clip_by_value_1:z:0*
T0*
_output_shapes
:m
(prune_low_magnitude_fc3_prun/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
&prune_low_magnitude_fc3_prun/truediv_3RealDiv&prune_low_magnitude_fc3_prun/mul_6:z:01prune_low_magnitude_fc3_prun/truediv_3/y:output:0*
T0*
_output_shapes
:i
$prune_low_magnitude_fc3_prun/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc3_prun/mul_7Mul-prune_low_magnitude_fc3_prun/mul_7/x:output:0*prune_low_magnitude_fc3_prun/truediv_3:z:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc3_prun/ReadVariableOp_4ReadVariableOp6prune_low_magnitude_fc3_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0�
"prune_low_magnitude_fc3_prun/Neg_3Neg5prune_low_magnitude_fc3_prun/ReadVariableOp_4:value:0*
T0*
_output_shapes
:�
"prune_low_magnitude_fc3_prun/add_6AddV2&prune_low_magnitude_fc3_prun/Neg_3:y:0&prune_low_magnitude_fc3_prun/mul_7:z:0*
T0*
_output_shapes
:i
$prune_low_magnitude_fc3_prun/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"prune_low_magnitude_fc3_prun/mul_8Mul-prune_low_magnitude_fc3_prun/mul_8/x:output:0&prune_low_magnitude_fc3_prun/add_6:z:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc3_prun/StopGradient_3StopGradient&prune_low_magnitude_fc3_prun/mul_8:z:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc3_prun/ReadVariableOp_5ReadVariableOp6prune_low_magnitude_fc3_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0�
"prune_low_magnitude_fc3_prun/add_7AddV25prune_low_magnitude_fc3_prun/ReadVariableOp_5:value:04prune_low_magnitude_fc3_prun/StopGradient_3:output:0*
T0*
_output_shapes
:�
$prune_low_magnitude_fc3_prun/BiasAddBiasAdd-prune_low_magnitude_fc3_prun/MatMul:product:0&prune_low_magnitude_fc3_prun/add_7:z:0*
T0*'
_output_shapes
:���������Q
relu3_enc/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :Q
relu3_enc/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :i
relu3_enc/PowPowrelu3_enc/Pow/x:output:0relu3_enc/Pow/y:output:0*
T0*
_output_shapes
: Y
relu3_enc/CastCastrelu3_enc/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: S
relu3_enc/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :S
relu3_enc/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : o
relu3_enc/Pow_1Powrelu3_enc/Pow_1/x:output:0relu3_enc/Pow_1/y:output:0*
T0*
_output_shapes
: ]
relu3_enc/Cast_1Castrelu3_enc/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: T
relu3_enc/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
relu3_enc/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : e
relu3_enc/Cast_2Castrelu3_enc/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
relu3_enc/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@e
relu3_enc/subSubrelu3_enc/Cast_2:y:0relu3_enc/sub/y:output:0*
T0*
_output_shapes
: d
relu3_enc/Pow_2Powrelu3_enc/Const:output:0relu3_enc/sub:z:0*
T0*
_output_shapes
: b
relu3_enc/sub_1Subrelu3_enc/Cast_1:y:0relu3_enc/Pow_2:z:0*
T0*
_output_shapes
: �
relu3_enc/LessEqual	LessEqual-prune_low_magnitude_fc3_prun/BiasAdd:output:0relu3_enc/sub_1:z:0*
T0*'
_output_shapes
:���������w
relu3_enc/ReluRelu-prune_low_magnitude_fc3_prun/BiasAdd:output:0*
T0*'
_output_shapes
:���������v
relu3_enc/ones_like/ShapeShape-prune_low_magnitude_fc3_prun/BiasAdd:output:0*
T0*
_output_shapes
:^
relu3_enc/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
relu3_enc/ones_likeFill"relu3_enc/ones_like/Shape:output:0"relu3_enc/ones_like/Const:output:0*
T0*'
_output_shapes
:���������b
relu3_enc/sub_2Subrelu3_enc/Cast_1:y:0relu3_enc/Pow_2:z:0*
T0*
_output_shapes
: y
relu3_enc/mulMulrelu3_enc/ones_like:output:0relu3_enc/sub_2:z:0*
T0*'
_output_shapes
:����������
relu3_enc/SelectV2SelectV2relu3_enc/LessEqual:z:0relu3_enc/Relu:activations:0relu3_enc/mul:z:0*
T0*'
_output_shapes
:����������
relu3_enc/mul_1Mul-prune_low_magnitude_fc3_prun/BiasAdd:output:0relu3_enc/Cast:y:0*
T0*'
_output_shapes
:���������y
relu3_enc/truedivRealDivrelu3_enc/mul_1:z:0relu3_enc/Cast_1:y:0*
T0*'
_output_shapes
:���������]
relu3_enc/NegNegrelu3_enc/truediv:z:0*
T0*'
_output_shapes
:���������a
relu3_enc/RoundRoundrelu3_enc/truediv:z:0*
T0*'
_output_shapes
:���������p
relu3_enc/addAddV2relu3_enc/Neg:y:0relu3_enc/Round:y:0*
T0*'
_output_shapes
:���������k
relu3_enc/StopGradientStopGradientrelu3_enc/add:z:0*
T0*'
_output_shapes
:����������
relu3_enc/add_1AddV2relu3_enc/truediv:z:0relu3_enc/StopGradient:output:0*
T0*'
_output_shapes
:���������y
relu3_enc/truediv_1RealDivrelu3_enc/add_1:z:0relu3_enc/Cast:y:0*
T0*'
_output_shapes
:���������Z
relu3_enc/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
relu3_enc/truediv_2RealDivrelu3_enc/truediv_2/x:output:0relu3_enc/Cast:y:0*
T0*
_output_shapes
: V
relu3_enc/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
relu3_enc/sub_3Subrelu3_enc/sub_3/x:output:0relu3_enc/truediv_2:z:0*
T0*
_output_shapes
: �
relu3_enc/clip_by_value/MinimumMinimumrelu3_enc/truediv_1:z:0relu3_enc/sub_3:z:0*
T0*'
_output_shapes
:���������^
relu3_enc/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
relu3_enc/clip_by_valueMaximum#relu3_enc/clip_by_value/Minimum:z:0"relu3_enc/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������{
relu3_enc/mul_2Mulrelu3_enc/Cast_1:y:0relu3_enc/clip_by_value:z:0*
T0*'
_output_shapes
:���������e
relu3_enc/Neg_1Negrelu3_enc/SelectV2:output:0*
T0*'
_output_shapes
:���������t
relu3_enc/add_2AddV2relu3_enc/Neg_1:y:0relu3_enc/mul_2:z:0*
T0*'
_output_shapes
:���������V
relu3_enc/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
relu3_enc/mul_3Mulrelu3_enc/mul_3/x:output:0relu3_enc/add_2:z:0*
T0*'
_output_shapes
:���������o
relu3_enc/StopGradient_1StopGradientrelu3_enc/mul_3:z:0*
T0*'
_output_shapes
:����������
relu3_enc/add_3AddV2relu3_enc/SelectV2:output:0!relu3_enc/StopGradient_1:output:0*
T0*'
_output_shapes
:����������
$encoder_output/MatMul/ReadVariableOpReadVariableOp-encoder_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
encoder_output/MatMulMatMulrelu3_enc/add_3:z:0,encoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%encoder_output/BiasAdd/ReadVariableOpReadVariableOp.encoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
encoder_output/BiasAddBiasAddencoder_output/MatMul:product:0-encoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
encoder_output/ReluReluencoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������K
-prune_low_magnitude_fc4_prunedclass/no_updateNoOp*
_output_shapes
 M
/prune_low_magnitude_fc4_prunedclass/no_update_1NoOp*
_output_shapes
 �
6prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOpReadVariableOp?prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
8prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1ReadVariableOpAprune_low_magnitude_fc4_prunedclass_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
'prune_low_magnitude_fc4_prunedclass/MulMul>prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp:value:0@prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
4prune_low_magnitude_fc4_prunedclass/AssignVariableOpAssignVariableOp?prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource+prune_low_magnitude_fc4_prunedclass/Mul:z:07^prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
.prune_low_magnitude_fc4_prunedclass/group_depsNoOp5^prune_low_magnitude_fc4_prunedclass/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0prune_low_magnitude_fc4_prunedclass/group_deps_1NoOp/^prune_low_magnitude_fc4_prunedclass/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 k
)prune_low_magnitude_fc4_prunedclass/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :k
)prune_low_magnitude_fc4_prunedclass/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
'prune_low_magnitude_fc4_prunedclass/PowPow2prune_low_magnitude_fc4_prunedclass/Pow/x:output:02prune_low_magnitude_fc4_prunedclass/Pow/y:output:0*
T0*
_output_shapes
: �
(prune_low_magnitude_fc4_prunedclass/CastCast+prune_low_magnitude_fc4_prunedclass/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
2prune_low_magnitude_fc4_prunedclass/ReadVariableOpReadVariableOp?prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource5^prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
_output_shapes

:*
dtype0p
+prune_low_magnitude_fc4_prunedclass/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
)prune_low_magnitude_fc4_prunedclass/mul_1Mul:prune_low_magnitude_fc4_prunedclass/ReadVariableOp:value:04prune_low_magnitude_fc4_prunedclass/mul_1/y:output:0*
T0*
_output_shapes

:�
+prune_low_magnitude_fc4_prunedclass/truedivRealDiv-prune_low_magnitude_fc4_prunedclass/mul_1:z:0,prune_low_magnitude_fc4_prunedclass/Cast:y:0*
T0*
_output_shapes

:�
'prune_low_magnitude_fc4_prunedclass/NegNeg/prune_low_magnitude_fc4_prunedclass/truediv:z:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc4_prunedclass/RoundRound/prune_low_magnitude_fc4_prunedclass/truediv:z:0*
T0*
_output_shapes

:�
'prune_low_magnitude_fc4_prunedclass/addAddV2+prune_low_magnitude_fc4_prunedclass/Neg:y:0-prune_low_magnitude_fc4_prunedclass/Round:y:0*
T0*
_output_shapes

:�
0prune_low_magnitude_fc4_prunedclass/StopGradientStopGradient+prune_low_magnitude_fc4_prunedclass/add:z:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc4_prunedclass/add_1AddV2/prune_low_magnitude_fc4_prunedclass/truediv:z:09prune_low_magnitude_fc4_prunedclass/StopGradient:output:0*
T0*
_output_shapes

:�
;prune_low_magnitude_fc4_prunedclass/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
9prune_low_magnitude_fc4_prunedclass/clip_by_value/MinimumMinimum-prune_low_magnitude_fc4_prunedclass/add_1:z:0Dprune_low_magnitude_fc4_prunedclass/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:x
3prune_low_magnitude_fc4_prunedclass/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
1prune_low_magnitude_fc4_prunedclass/clip_by_valueMaximum=prune_low_magnitude_fc4_prunedclass/clip_by_value/Minimum:z:0<prune_low_magnitude_fc4_prunedclass/clip_by_value/y:output:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc4_prunedclass/mul_2Mul,prune_low_magnitude_fc4_prunedclass/Cast:y:05prune_low_magnitude_fc4_prunedclass/clip_by_value:z:0*
T0*
_output_shapes

:t
/prune_low_magnitude_fc4_prunedclass/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
-prune_low_magnitude_fc4_prunedclass/truediv_1RealDiv-prune_low_magnitude_fc4_prunedclass/mul_2:z:08prune_low_magnitude_fc4_prunedclass/truediv_1/y:output:0*
T0*
_output_shapes

:p
+prune_low_magnitude_fc4_prunedclass/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)prune_low_magnitude_fc4_prunedclass/mul_3Mul4prune_low_magnitude_fc4_prunedclass/mul_3/x:output:01prune_low_magnitude_fc4_prunedclass/truediv_1:z:0*
T0*
_output_shapes

:�
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1ReadVariableOp?prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource5^prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
_output_shapes

:*
dtype0�
)prune_low_magnitude_fc4_prunedclass/Neg_1Neg<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
)prune_low_magnitude_fc4_prunedclass/add_2AddV2-prune_low_magnitude_fc4_prunedclass/Neg_1:y:0-prune_low_magnitude_fc4_prunedclass/mul_3:z:0*
T0*
_output_shapes

:p
+prune_low_magnitude_fc4_prunedclass/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)prune_low_magnitude_fc4_prunedclass/mul_4Mul4prune_low_magnitude_fc4_prunedclass/mul_4/x:output:0-prune_low_magnitude_fc4_prunedclass/add_2:z:0*
T0*
_output_shapes

:�
2prune_low_magnitude_fc4_prunedclass/StopGradient_1StopGradient-prune_low_magnitude_fc4_prunedclass/mul_4:z:0*
T0*
_output_shapes

:�
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2ReadVariableOp?prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource5^prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
_output_shapes

:*
dtype0�
)prune_low_magnitude_fc4_prunedclass/add_3AddV2<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2:value:0;prune_low_magnitude_fc4_prunedclass/StopGradient_1:output:0*
T0*
_output_shapes

:�
*prune_low_magnitude_fc4_prunedclass/MatMulMatMul!encoder_output/Relu:activations:0-prune_low_magnitude_fc4_prunedclass/add_3:z:0*
T0*'
_output_shapes
:���������m
+prune_low_magnitude_fc4_prunedclass/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :m
+prune_low_magnitude_fc4_prunedclass/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
)prune_low_magnitude_fc4_prunedclass/Pow_1Pow4prune_low_magnitude_fc4_prunedclass/Pow_1/x:output:04prune_low_magnitude_fc4_prunedclass/Pow_1/y:output:0*
T0*
_output_shapes
: �
*prune_low_magnitude_fc4_prunedclass/Cast_1Cast-prune_low_magnitude_fc4_prunedclass/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3ReadVariableOp=prune_low_magnitude_fc4_prunedclass_readvariableop_3_resource*
_output_shapes
:*
dtype0p
+prune_low_magnitude_fc4_prunedclass/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
)prune_low_magnitude_fc4_prunedclass/mul_5Mul<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3:value:04prune_low_magnitude_fc4_prunedclass/mul_5/y:output:0*
T0*
_output_shapes
:�
-prune_low_magnitude_fc4_prunedclass/truediv_2RealDiv-prune_low_magnitude_fc4_prunedclass/mul_5:z:0.prune_low_magnitude_fc4_prunedclass/Cast_1:y:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/Neg_2Neg1prune_low_magnitude_fc4_prunedclass/truediv_2:z:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc4_prunedclass/Round_1Round1prune_low_magnitude_fc4_prunedclass/truediv_2:z:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/add_4AddV2-prune_low_magnitude_fc4_prunedclass/Neg_2:y:0/prune_low_magnitude_fc4_prunedclass/Round_1:y:0*
T0*
_output_shapes
:�
2prune_low_magnitude_fc4_prunedclass/StopGradient_2StopGradient-prune_low_magnitude_fc4_prunedclass/add_4:z:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/add_5AddV21prune_low_magnitude_fc4_prunedclass/truediv_2:z:0;prune_low_magnitude_fc4_prunedclass/StopGradient_2:output:0*
T0*
_output_shapes
:�
=prune_low_magnitude_fc4_prunedclass/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
;prune_low_magnitude_fc4_prunedclass/clip_by_value_1/MinimumMinimum-prune_low_magnitude_fc4_prunedclass/add_5:z:0Fprune_low_magnitude_fc4_prunedclass/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:z
5prune_low_magnitude_fc4_prunedclass/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
3prune_low_magnitude_fc4_prunedclass/clip_by_value_1Maximum?prune_low_magnitude_fc4_prunedclass/clip_by_value_1/Minimum:z:0>prune_low_magnitude_fc4_prunedclass/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/mul_6Mul.prune_low_magnitude_fc4_prunedclass/Cast_1:y:07prune_low_magnitude_fc4_prunedclass/clip_by_value_1:z:0*
T0*
_output_shapes
:t
/prune_low_magnitude_fc4_prunedclass/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
-prune_low_magnitude_fc4_prunedclass/truediv_3RealDiv-prune_low_magnitude_fc4_prunedclass/mul_6:z:08prune_low_magnitude_fc4_prunedclass/truediv_3/y:output:0*
T0*
_output_shapes
:p
+prune_low_magnitude_fc4_prunedclass/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)prune_low_magnitude_fc4_prunedclass/mul_7Mul4prune_low_magnitude_fc4_prunedclass/mul_7/x:output:01prune_low_magnitude_fc4_prunedclass/truediv_3:z:0*
T0*
_output_shapes
:�
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4ReadVariableOp=prune_low_magnitude_fc4_prunedclass_readvariableop_3_resource*
_output_shapes
:*
dtype0�
)prune_low_magnitude_fc4_prunedclass/Neg_3Neg<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4:value:0*
T0*
_output_shapes
:�
)prune_low_magnitude_fc4_prunedclass/add_6AddV2-prune_low_magnitude_fc4_prunedclass/Neg_3:y:0-prune_low_magnitude_fc4_prunedclass/mul_7:z:0*
T0*
_output_shapes
:p
+prune_low_magnitude_fc4_prunedclass/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)prune_low_magnitude_fc4_prunedclass/mul_8Mul4prune_low_magnitude_fc4_prunedclass/mul_8/x:output:0-prune_low_magnitude_fc4_prunedclass/add_6:z:0*
T0*
_output_shapes
:�
2prune_low_magnitude_fc4_prunedclass/StopGradient_3StopGradient-prune_low_magnitude_fc4_prunedclass/mul_8:z:0*
T0*
_output_shapes
:�
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5ReadVariableOp=prune_low_magnitude_fc4_prunedclass_readvariableop_3_resource*
_output_shapes
:*
dtype0�
)prune_low_magnitude_fc4_prunedclass/add_7AddV2<prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5:value:0;prune_low_magnitude_fc4_prunedclass/StopGradient_3:output:0*
T0*
_output_shapes
:�
+prune_low_magnitude_fc4_prunedclass/BiasAddBiasAdd4prune_low_magnitude_fc4_prunedclass/MatMul:product:0-prune_low_magnitude_fc4_prunedclass/add_7:z:0*
T0*'
_output_shapes
:���������W
prunclass_relu4/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :W
prunclass_relu4/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :{
prunclass_relu4/PowPowprunclass_relu4/Pow/x:output:0prunclass_relu4/Pow/y:output:0*
T0*
_output_shapes
: e
prunclass_relu4/CastCastprunclass_relu4/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: Y
prunclass_relu4/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
prunclass_relu4/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
prunclass_relu4/Pow_1Pow prunclass_relu4/Pow_1/x:output:0 prunclass_relu4/Pow_1/y:output:0*
T0*
_output_shapes
: i
prunclass_relu4/Cast_1Castprunclass_relu4/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
prunclass_relu4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
prunclass_relu4/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : q
prunclass_relu4/Cast_2Cast!prunclass_relu4/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
prunclass_relu4/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
prunclass_relu4/subSubprunclass_relu4/Cast_2:y:0prunclass_relu4/sub/y:output:0*
T0*
_output_shapes
: v
prunclass_relu4/Pow_2Powprunclass_relu4/Const:output:0prunclass_relu4/sub:z:0*
T0*
_output_shapes
: t
prunclass_relu4/sub_1Subprunclass_relu4/Cast_1:y:0prunclass_relu4/Pow_2:z:0*
T0*
_output_shapes
: �
prunclass_relu4/LessEqual	LessEqual4prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0prunclass_relu4/sub_1:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/ReluRelu4prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0*
T0*'
_output_shapes
:����������
prunclass_relu4/ones_like/ShapeShape4prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0*
T0*
_output_shapes
:d
prunclass_relu4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
prunclass_relu4/ones_likeFill(prunclass_relu4/ones_like/Shape:output:0(prunclass_relu4/ones_like/Const:output:0*
T0*'
_output_shapes
:���������t
prunclass_relu4/sub_2Subprunclass_relu4/Cast_1:y:0prunclass_relu4/Pow_2:z:0*
T0*
_output_shapes
: �
prunclass_relu4/mulMul"prunclass_relu4/ones_like:output:0prunclass_relu4/sub_2:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/SelectV2SelectV2prunclass_relu4/LessEqual:z:0"prunclass_relu4/Relu:activations:0prunclass_relu4/mul:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/mul_1Mul4prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0prunclass_relu4/Cast:y:0*
T0*'
_output_shapes
:����������
prunclass_relu4/truedivRealDivprunclass_relu4/mul_1:z:0prunclass_relu4/Cast_1:y:0*
T0*'
_output_shapes
:���������i
prunclass_relu4/NegNegprunclass_relu4/truediv:z:0*
T0*'
_output_shapes
:���������m
prunclass_relu4/RoundRoundprunclass_relu4/truediv:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/addAddV2prunclass_relu4/Neg:y:0prunclass_relu4/Round:y:0*
T0*'
_output_shapes
:���������w
prunclass_relu4/StopGradientStopGradientprunclass_relu4/add:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/add_1AddV2prunclass_relu4/truediv:z:0%prunclass_relu4/StopGradient:output:0*
T0*'
_output_shapes
:����������
prunclass_relu4/truediv_1RealDivprunclass_relu4/add_1:z:0prunclass_relu4/Cast:y:0*
T0*'
_output_shapes
:���������`
prunclass_relu4/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
prunclass_relu4/truediv_2RealDiv$prunclass_relu4/truediv_2/x:output:0prunclass_relu4/Cast:y:0*
T0*
_output_shapes
: \
prunclass_relu4/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
prunclass_relu4/sub_3Sub prunclass_relu4/sub_3/x:output:0prunclass_relu4/truediv_2:z:0*
T0*
_output_shapes
: �
%prunclass_relu4/clip_by_value/MinimumMinimumprunclass_relu4/truediv_1:z:0prunclass_relu4/sub_3:z:0*
T0*'
_output_shapes
:���������d
prunclass_relu4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
prunclass_relu4/clip_by_valueMaximum)prunclass_relu4/clip_by_value/Minimum:z:0(prunclass_relu4/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
prunclass_relu4/mul_2Mulprunclass_relu4/Cast_1:y:0!prunclass_relu4/clip_by_value:z:0*
T0*'
_output_shapes
:���������q
prunclass_relu4/Neg_1Neg!prunclass_relu4/SelectV2:output:0*
T0*'
_output_shapes
:����������
prunclass_relu4/add_2AddV2prunclass_relu4/Neg_1:y:0prunclass_relu4/mul_2:z:0*
T0*'
_output_shapes
:���������\
prunclass_relu4/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
prunclass_relu4/mul_3Mul prunclass_relu4/mul_3/x:output:0prunclass_relu4/add_2:z:0*
T0*'
_output_shapes
:���������{
prunclass_relu4/StopGradient_1StopGradientprunclass_relu4/mul_3:z:0*
T0*'
_output_shapes
:����������
prunclass_relu4/add_3AddV2!prunclass_relu4/SelectV2:output:0'prunclass_relu4/StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
fc5_class/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :Q
fc5_class/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : i
fc5_class/PowPowfc5_class/Pow/x:output:0fc5_class/Pow/y:output:0*
T0*
_output_shapes
: Y
fc5_class/CastCastfc5_class/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: z
fc5_class/ReadVariableOpReadVariableOp!fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0T
fc5_class/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   By
fc5_class/mulMul fc5_class/ReadVariableOp:value:0fc5_class/mul/y:output:0*
T0*
_output_shapes

:(l
fc5_class/truedivRealDivfc5_class/mul:z:0fc5_class/Cast:y:0*
T0*
_output_shapes

:(T
fc5_class/NegNegfc5_class/truediv:z:0*
T0*
_output_shapes

:(X
fc5_class/RoundRoundfc5_class/truediv:z:0*
T0*
_output_shapes

:(g
fc5_class/addAddV2fc5_class/Neg:y:0fc5_class/Round:y:0*
T0*
_output_shapes

:(b
fc5_class/StopGradientStopGradientfc5_class/add:z:0*
T0*
_output_shapes

:(y
fc5_class/add_1AddV2fc5_class/truediv:z:0fc5_class/StopGradient:output:0*
T0*
_output_shapes

:(f
!fc5_class/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
fc5_class/clip_by_value/MinimumMinimumfc5_class/add_1:z:0*fc5_class/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(^
fc5_class/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
fc5_class/clip_by_valueMaximum#fc5_class/clip_by_value/Minimum:z:0"fc5_class/clip_by_value/y:output:0*
T0*
_output_shapes

:(p
fc5_class/mul_1Mulfc5_class/Cast:y:0fc5_class/clip_by_value:z:0*
T0*
_output_shapes

:(Z
fc5_class/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B|
fc5_class/truediv_1RealDivfc5_class/mul_1:z:0fc5_class/truediv_1/y:output:0*
T0*
_output_shapes

:(V
fc5_class/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
fc5_class/mul_2Mulfc5_class/mul_2/x:output:0fc5_class/truediv_1:z:0*
T0*
_output_shapes

:(|
fc5_class/ReadVariableOp_1ReadVariableOp!fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0c
fc5_class/Neg_1Neg"fc5_class/ReadVariableOp_1:value:0*
T0*
_output_shapes

:(k
fc5_class/add_2AddV2fc5_class/Neg_1:y:0fc5_class/mul_2:z:0*
T0*
_output_shapes

:(V
fc5_class/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?p
fc5_class/mul_3Mulfc5_class/mul_3/x:output:0fc5_class/add_2:z:0*
T0*
_output_shapes

:(f
fc5_class/StopGradient_1StopGradientfc5_class/mul_3:z:0*
T0*
_output_shapes

:(|
fc5_class/ReadVariableOp_2ReadVariableOp!fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0�
fc5_class/add_3AddV2"fc5_class/ReadVariableOp_2:value:0!fc5_class/StopGradient_1:output:0*
T0*
_output_shapes

:(|
fc5_class/MatMulMatMulprunclass_relu4/add_3:z:0fc5_class/add_3:z:0*
T0*'
_output_shapes
:���������(S
fc5_class/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :S
fc5_class/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : o
fc5_class/Pow_1Powfc5_class/Pow_1/x:output:0fc5_class/Pow_1/y:output:0*
T0*
_output_shapes
: ]
fc5_class/Cast_1Castfc5_class/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: z
fc5_class/ReadVariableOp_3ReadVariableOp#fc5_class_readvariableop_3_resource*
_output_shapes
:(*
dtype0V
fc5_class/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B{
fc5_class/mul_4Mul"fc5_class/ReadVariableOp_3:value:0fc5_class/mul_4/y:output:0*
T0*
_output_shapes
:(n
fc5_class/truediv_2RealDivfc5_class/mul_4:z:0fc5_class/Cast_1:y:0*
T0*
_output_shapes
:(T
fc5_class/Neg_2Negfc5_class/truediv_2:z:0*
T0*
_output_shapes
:(X
fc5_class/Round_1Roundfc5_class/truediv_2:z:0*
T0*
_output_shapes
:(i
fc5_class/add_4AddV2fc5_class/Neg_2:y:0fc5_class/Round_1:y:0*
T0*
_output_shapes
:(b
fc5_class/StopGradient_2StopGradientfc5_class/add_4:z:0*
T0*
_output_shapes
:(y
fc5_class/add_5AddV2fc5_class/truediv_2:z:0!fc5_class/StopGradient_2:output:0*
T0*
_output_shapes
:(h
#fc5_class/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
!fc5_class/clip_by_value_1/MinimumMinimumfc5_class/add_5:z:0,fc5_class/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(`
fc5_class/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
fc5_class/clip_by_value_1Maximum%fc5_class/clip_by_value_1/Minimum:z:0$fc5_class/clip_by_value_1/y:output:0*
T0*
_output_shapes
:(p
fc5_class/mul_5Mulfc5_class/Cast_1:y:0fc5_class/clip_by_value_1:z:0*
T0*
_output_shapes
:(Z
fc5_class/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bx
fc5_class/truediv_3RealDivfc5_class/mul_5:z:0fc5_class/truediv_3/y:output:0*
T0*
_output_shapes
:(V
fc5_class/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?p
fc5_class/mul_6Mulfc5_class/mul_6/x:output:0fc5_class/truediv_3:z:0*
T0*
_output_shapes
:(z
fc5_class/ReadVariableOp_4ReadVariableOp#fc5_class_readvariableop_3_resource*
_output_shapes
:(*
dtype0_
fc5_class/Neg_3Neg"fc5_class/ReadVariableOp_4:value:0*
T0*
_output_shapes
:(g
fc5_class/add_6AddV2fc5_class/Neg_3:y:0fc5_class/mul_6:z:0*
T0*
_output_shapes
:(V
fc5_class/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
fc5_class/mul_7Mulfc5_class/mul_7/x:output:0fc5_class/add_6:z:0*
T0*
_output_shapes
:(b
fc5_class/StopGradient_3StopGradientfc5_class/mul_7:z:0*
T0*
_output_shapes
:(z
fc5_class/ReadVariableOp_5ReadVariableOp#fc5_class_readvariableop_3_resource*
_output_shapes
:(*
dtype0�
fc5_class/add_7AddV2"fc5_class/ReadVariableOp_5:value:0!fc5_class/StopGradient_3:output:0*
T0*
_output_shapes
:(
fc5_class/BiasAddBiasAddfc5_class/MatMul:product:0fc5_class/add_7:z:0*
T0*'
_output_shapes
:���������(S
class_relu5/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :S
class_relu5/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :o
class_relu5/PowPowclass_relu5/Pow/x:output:0class_relu5/Pow/y:output:0*
T0*
_output_shapes
: ]
class_relu5/CastCastclass_relu5/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: U
class_relu5/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :U
class_relu5/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : u
class_relu5/Pow_1Powclass_relu5/Pow_1/x:output:0class_relu5/Pow_1/y:output:0*
T0*
_output_shapes
: a
class_relu5/Cast_1Castclass_relu5/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: V
class_relu5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @V
class_relu5/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : i
class_relu5/Cast_2Castclass_relu5/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: V
class_relu5/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@k
class_relu5/subSubclass_relu5/Cast_2:y:0class_relu5/sub/y:output:0*
T0*
_output_shapes
: j
class_relu5/Pow_2Powclass_relu5/Const:output:0class_relu5/sub:z:0*
T0*
_output_shapes
: h
class_relu5/sub_1Subclass_relu5/Cast_1:y:0class_relu5/Pow_2:z:0*
T0*
_output_shapes
: �
class_relu5/LessEqual	LessEqualfc5_class/BiasAdd:output:0class_relu5/sub_1:z:0*
T0*'
_output_shapes
:���������(f
class_relu5/ReluRelufc5_class/BiasAdd:output:0*
T0*'
_output_shapes
:���������(e
class_relu5/ones_like/ShapeShapefc5_class/BiasAdd:output:0*
T0*
_output_shapes
:`
class_relu5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
class_relu5/ones_likeFill$class_relu5/ones_like/Shape:output:0$class_relu5/ones_like/Const:output:0*
T0*'
_output_shapes
:���������(h
class_relu5/sub_2Subclass_relu5/Cast_1:y:0class_relu5/Pow_2:z:0*
T0*
_output_shapes
: 
class_relu5/mulMulclass_relu5/ones_like:output:0class_relu5/sub_2:z:0*
T0*'
_output_shapes
:���������(�
class_relu5/SelectV2SelectV2class_relu5/LessEqual:z:0class_relu5/Relu:activations:0class_relu5/mul:z:0*
T0*'
_output_shapes
:���������(|
class_relu5/mul_1Mulfc5_class/BiasAdd:output:0class_relu5/Cast:y:0*
T0*'
_output_shapes
:���������(
class_relu5/truedivRealDivclass_relu5/mul_1:z:0class_relu5/Cast_1:y:0*
T0*'
_output_shapes
:���������(a
class_relu5/NegNegclass_relu5/truediv:z:0*
T0*'
_output_shapes
:���������(e
class_relu5/RoundRoundclass_relu5/truediv:z:0*
T0*'
_output_shapes
:���������(v
class_relu5/addAddV2class_relu5/Neg:y:0class_relu5/Round:y:0*
T0*'
_output_shapes
:���������(o
class_relu5/StopGradientStopGradientclass_relu5/add:z:0*
T0*'
_output_shapes
:���������(�
class_relu5/add_1AddV2class_relu5/truediv:z:0!class_relu5/StopGradient:output:0*
T0*'
_output_shapes
:���������(
class_relu5/truediv_1RealDivclass_relu5/add_1:z:0class_relu5/Cast:y:0*
T0*'
_output_shapes
:���������(\
class_relu5/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
class_relu5/truediv_2RealDiv class_relu5/truediv_2/x:output:0class_relu5/Cast:y:0*
T0*
_output_shapes
: X
class_relu5/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?r
class_relu5/sub_3Subclass_relu5/sub_3/x:output:0class_relu5/truediv_2:z:0*
T0*
_output_shapes
: �
!class_relu5/clip_by_value/MinimumMinimumclass_relu5/truediv_1:z:0class_relu5/sub_3:z:0*
T0*'
_output_shapes
:���������(`
class_relu5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
class_relu5/clip_by_valueMaximum%class_relu5/clip_by_value/Minimum:z:0$class_relu5/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������(�
class_relu5/mul_2Mulclass_relu5/Cast_1:y:0class_relu5/clip_by_value:z:0*
T0*'
_output_shapes
:���������(i
class_relu5/Neg_1Negclass_relu5/SelectV2:output:0*
T0*'
_output_shapes
:���������(z
class_relu5/add_2AddV2class_relu5/Neg_1:y:0class_relu5/mul_2:z:0*
T0*'
_output_shapes
:���������(X
class_relu5/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
class_relu5/mul_3Mulclass_relu5/mul_3/x:output:0class_relu5/add_2:z:0*
T0*'
_output_shapes
:���������(s
class_relu5/StopGradient_1StopGradientclass_relu5/mul_3:z:0*
T0*'
_output_shapes
:���������(�
class_relu5/add_3AddV2class_relu5/SelectV2:output:0#class_relu5/StopGradient_1:output:0*
T0*'
_output_shapes
:���������(V
classifier_out/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :V
classifier_out/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :x
classifier_out/PowPowclassifier_out/Pow/x:output:0classifier_out/Pow/y:output:0*
T0*
_output_shapes
: c
classifier_out/CastCastclassifier_out/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
classifier_out/ReadVariableOpReadVariableOp&classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0Y
classifier_out/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier_out/mulMul%classifier_out/ReadVariableOp:value:0classifier_out/mul/y:output:0*
T0*
_output_shapes

:(
{
classifier_out/truedivRealDivclassifier_out/mul:z:0classifier_out/Cast:y:0*
T0*
_output_shapes

:(
^
classifier_out/NegNegclassifier_out/truediv:z:0*
T0*
_output_shapes

:(
b
classifier_out/RoundRoundclassifier_out/truediv:z:0*
T0*
_output_shapes

:(
v
classifier_out/addAddV2classifier_out/Neg:y:0classifier_out/Round:y:0*
T0*
_output_shapes

:(
l
classifier_out/StopGradientStopGradientclassifier_out/add:z:0*
T0*
_output_shapes

:(
�
classifier_out/add_1AddV2classifier_out/truediv:z:0$classifier_out/StopGradient:output:0*
T0*
_output_shapes

:(
k
&classifier_out/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��F�
$classifier_out/clip_by_value/MinimumMinimumclassifier_out/add_1:z:0/classifier_out/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(
c
classifier_out/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
classifier_out/clip_by_valueMaximum(classifier_out/clip_by_value/Minimum:z:0'classifier_out/clip_by_value/y:output:0*
T0*
_output_shapes

:(

classifier_out/mul_1Mulclassifier_out/Cast:y:0 classifier_out/clip_by_value:z:0*
T0*
_output_shapes

:(
_
classifier_out/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier_out/truediv_1RealDivclassifier_out/mul_1:z:0#classifier_out/truediv_1/y:output:0*
T0*
_output_shapes

:(
[
classifier_out/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier_out/mul_2Mulclassifier_out/mul_2/x:output:0classifier_out/truediv_1:z:0*
T0*
_output_shapes

:(
�
classifier_out/ReadVariableOp_1ReadVariableOp&classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0m
classifier_out/Neg_1Neg'classifier_out/ReadVariableOp_1:value:0*
T0*
_output_shapes

:(
z
classifier_out/add_2AddV2classifier_out/Neg_1:y:0classifier_out/mul_2:z:0*
T0*
_output_shapes

:(
[
classifier_out/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
classifier_out/mul_3Mulclassifier_out/mul_3/x:output:0classifier_out/add_2:z:0*
T0*
_output_shapes

:(
p
classifier_out/StopGradient_1StopGradientclassifier_out/mul_3:z:0*
T0*
_output_shapes

:(
�
classifier_out/ReadVariableOp_2ReadVariableOp&classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0�
classifier_out/add_3AddV2'classifier_out/ReadVariableOp_2:value:0&classifier_out/StopGradient_1:output:0*
T0*
_output_shapes

:(
�
classifier_out/MatMulMatMulclass_relu5/add_3:z:0classifier_out/add_3:z:0*
T0*'
_output_shapes
:���������
X
classifier_out/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :X
classifier_out/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
classifier_out/Pow_1Powclassifier_out/Pow_1/x:output:0classifier_out/Pow_1/y:output:0*
T0*
_output_shapes
: g
classifier_out/Cast_1Castclassifier_out/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
classifier_out/ReadVariableOp_3ReadVariableOp(classifier_out_readvariableop_3_resource*
_output_shapes
:
*
dtype0[
classifier_out/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier_out/mul_4Mul'classifier_out/ReadVariableOp_3:value:0classifier_out/mul_4/y:output:0*
T0*
_output_shapes
:
}
classifier_out/truediv_2RealDivclassifier_out/mul_4:z:0classifier_out/Cast_1:y:0*
T0*
_output_shapes
:
^
classifier_out/Neg_2Negclassifier_out/truediv_2:z:0*
T0*
_output_shapes
:
b
classifier_out/Round_1Roundclassifier_out/truediv_2:z:0*
T0*
_output_shapes
:
x
classifier_out/add_4AddV2classifier_out/Neg_2:y:0classifier_out/Round_1:y:0*
T0*
_output_shapes
:
l
classifier_out/StopGradient_2StopGradientclassifier_out/add_4:z:0*
T0*
_output_shapes
:
�
classifier_out/add_5AddV2classifier_out/truediv_2:z:0&classifier_out/StopGradient_2:output:0*
T0*
_output_shapes
:
m
(classifier_out/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��F�
&classifier_out/clip_by_value_1/MinimumMinimumclassifier_out/add_5:z:01classifier_out/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:
e
 classifier_out/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
classifier_out/clip_by_value_1Maximum*classifier_out/clip_by_value_1/Minimum:z:0)classifier_out/clip_by_value_1/y:output:0*
T0*
_output_shapes
:

classifier_out/mul_5Mulclassifier_out/Cast_1:y:0"classifier_out/clip_by_value_1:z:0*
T0*
_output_shapes
:
_
classifier_out/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier_out/truediv_3RealDivclassifier_out/mul_5:z:0#classifier_out/truediv_3/y:output:0*
T0*
_output_shapes
:
[
classifier_out/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
classifier_out/mul_6Mulclassifier_out/mul_6/x:output:0classifier_out/truediv_3:z:0*
T0*
_output_shapes
:
�
classifier_out/ReadVariableOp_4ReadVariableOp(classifier_out_readvariableop_3_resource*
_output_shapes
:
*
dtype0i
classifier_out/Neg_3Neg'classifier_out/ReadVariableOp_4:value:0*
T0*
_output_shapes
:
v
classifier_out/add_6AddV2classifier_out/Neg_3:y:0classifier_out/mul_6:z:0*
T0*
_output_shapes
:
[
classifier_out/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{
classifier_out/mul_7Mulclassifier_out/mul_7/x:output:0classifier_out/add_6:z:0*
T0*
_output_shapes
:
l
classifier_out/StopGradient_3StopGradientclassifier_out/mul_7:z:0*
T0*
_output_shapes
:
�
classifier_out/ReadVariableOp_5ReadVariableOp(classifier_out_readvariableop_3_resource*
_output_shapes
:
*
dtype0�
classifier_out/add_7AddV2'classifier_out/ReadVariableOp_5:value:0&classifier_out/StopGradient_3:output:0*
T0*
_output_shapes
:
�
classifier_out/BiasAddBiasAddclassifier_out/MatMul:product:0classifier_out/add_7:z:0*
T0*'
_output_shapes
:���������
w
classifier_output/SoftmaxSoftmaxclassifier_out/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp?prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource5^prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
/fc5_class/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0�
 fc5_class/kernel/Regularizer/AbsAbs7fc5_class/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(s
"fc5_class/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 fc5_class/kernel/Regularizer/SumSum$fc5_class/kernel/Regularizer/Abs:y:0+fc5_class/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"fc5_class/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
 fc5_class/kernel/Regularizer/mulMul+fc5_class/kernel/Regularizer/mul/x:output:0)fc5_class/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4classifier_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0�
%classifier_out/kernel/Regularizer/AbsAbs<classifier_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(
x
'classifier_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%classifier_out/kernel/Regularizer/SumSum)classifier_out/kernel/Regularizer/Abs:y:00classifier_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'classifier_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%classifier_out/kernel/Regularizer/mulMul0classifier_out/kernel/Regularizer/mul/x:output:0.classifier_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#classifier_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^classifier_out/ReadVariableOp ^classifier_out/ReadVariableOp_1 ^classifier_out/ReadVariableOp_2 ^classifier_out/ReadVariableOp_3 ^classifier_out/ReadVariableOp_4 ^classifier_out/ReadVariableOp_55^classifier_out/kernel/Regularizer/Abs/ReadVariableOp&^encoder_output/BiasAdd/ReadVariableOp%^encoder_output/MatMul/ReadVariableOp^fc1/ReadVariableOp^fc1/ReadVariableOp_1^fc1/ReadVariableOp_2^fc1/ReadVariableOp_3^fc1/ReadVariableOp_4^fc1/ReadVariableOp_5^fc5_class/ReadVariableOp^fc5_class/ReadVariableOp_1^fc5_class/ReadVariableOp_2^fc5_class/ReadVariableOp_3^fc5_class/ReadVariableOp_4^fc5_class/ReadVariableOp_50^fc5_class/kernel/Regularizer/Abs/ReadVariableOp.^prune_low_magnitude_fc2_prun/AssignVariableOp0^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1,^prune_low_magnitude_fc2_prun/ReadVariableOp.^prune_low_magnitude_fc2_prun/ReadVariableOp_1.^prune_low_magnitude_fc2_prun/ReadVariableOp_2.^prune_low_magnitude_fc2_prun/ReadVariableOp_3.^prune_low_magnitude_fc2_prun/ReadVariableOp_4.^prune_low_magnitude_fc2_prun/ReadVariableOp_5.^prune_low_magnitude_fc3_prun/AssignVariableOp0^prune_low_magnitude_fc3_prun/Mul/ReadVariableOp2^prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1,^prune_low_magnitude_fc3_prun/ReadVariableOp.^prune_low_magnitude_fc3_prun/ReadVariableOp_1.^prune_low_magnitude_fc3_prun/ReadVariableOp_2.^prune_low_magnitude_fc3_prun/ReadVariableOp_3.^prune_low_magnitude_fc3_prun/ReadVariableOp_4.^prune_low_magnitude_fc3_prun/ReadVariableOp_55^prune_low_magnitude_fc4_prunedclass/AssignVariableOp7^prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp9^prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_13^prune_low_magnitude_fc4_prunedclass/ReadVariableOp5^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_15^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_25^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_35^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_45^prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5J^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������@: : : : : : : : : : : : : : : : : 2>
classifier_out/ReadVariableOpclassifier_out/ReadVariableOp2B
classifier_out/ReadVariableOp_1classifier_out/ReadVariableOp_12B
classifier_out/ReadVariableOp_2classifier_out/ReadVariableOp_22B
classifier_out/ReadVariableOp_3classifier_out/ReadVariableOp_32B
classifier_out/ReadVariableOp_4classifier_out/ReadVariableOp_42B
classifier_out/ReadVariableOp_5classifier_out/ReadVariableOp_52l
4classifier_out/kernel/Regularizer/Abs/ReadVariableOp4classifier_out/kernel/Regularizer/Abs/ReadVariableOp2N
%encoder_output/BiasAdd/ReadVariableOp%encoder_output/BiasAdd/ReadVariableOp2L
$encoder_output/MatMul/ReadVariableOp$encoder_output/MatMul/ReadVariableOp2(
fc1/ReadVariableOpfc1/ReadVariableOp2,
fc1/ReadVariableOp_1fc1/ReadVariableOp_12,
fc1/ReadVariableOp_2fc1/ReadVariableOp_22,
fc1/ReadVariableOp_3fc1/ReadVariableOp_32,
fc1/ReadVariableOp_4fc1/ReadVariableOp_42,
fc1/ReadVariableOp_5fc1/ReadVariableOp_524
fc5_class/ReadVariableOpfc5_class/ReadVariableOp28
fc5_class/ReadVariableOp_1fc5_class/ReadVariableOp_128
fc5_class/ReadVariableOp_2fc5_class/ReadVariableOp_228
fc5_class/ReadVariableOp_3fc5_class/ReadVariableOp_328
fc5_class/ReadVariableOp_4fc5_class/ReadVariableOp_428
fc5_class/ReadVariableOp_5fc5_class/ReadVariableOp_52b
/fc5_class/kernel/Regularizer/Abs/ReadVariableOp/fc5_class/kernel/Regularizer/Abs/ReadVariableOp2^
-prune_low_magnitude_fc2_prun/AssignVariableOp-prune_low_magnitude_fc2_prun/AssignVariableOp2b
/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2f
1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_11prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_12Z
+prune_low_magnitude_fc2_prun/ReadVariableOp+prune_low_magnitude_fc2_prun/ReadVariableOp2^
-prune_low_magnitude_fc2_prun/ReadVariableOp_1-prune_low_magnitude_fc2_prun/ReadVariableOp_12^
-prune_low_magnitude_fc2_prun/ReadVariableOp_2-prune_low_magnitude_fc2_prun/ReadVariableOp_22^
-prune_low_magnitude_fc2_prun/ReadVariableOp_3-prune_low_magnitude_fc2_prun/ReadVariableOp_32^
-prune_low_magnitude_fc2_prun/ReadVariableOp_4-prune_low_magnitude_fc2_prun/ReadVariableOp_42^
-prune_low_magnitude_fc2_prun/ReadVariableOp_5-prune_low_magnitude_fc2_prun/ReadVariableOp_52^
-prune_low_magnitude_fc3_prun/AssignVariableOp-prune_low_magnitude_fc3_prun/AssignVariableOp2b
/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp2f
1prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_11prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_12Z
+prune_low_magnitude_fc3_prun/ReadVariableOp+prune_low_magnitude_fc3_prun/ReadVariableOp2^
-prune_low_magnitude_fc3_prun/ReadVariableOp_1-prune_low_magnitude_fc3_prun/ReadVariableOp_12^
-prune_low_magnitude_fc3_prun/ReadVariableOp_2-prune_low_magnitude_fc3_prun/ReadVariableOp_22^
-prune_low_magnitude_fc3_prun/ReadVariableOp_3-prune_low_magnitude_fc3_prun/ReadVariableOp_32^
-prune_low_magnitude_fc3_prun/ReadVariableOp_4-prune_low_magnitude_fc3_prun/ReadVariableOp_42^
-prune_low_magnitude_fc3_prun/ReadVariableOp_5-prune_low_magnitude_fc3_prun/ReadVariableOp_52l
4prune_low_magnitude_fc4_prunedclass/AssignVariableOp4prune_low_magnitude_fc4_prunedclass/AssignVariableOp2p
6prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp6prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp2t
8prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_18prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_12h
2prune_low_magnitude_fc4_prunedclass/ReadVariableOp2prune_low_magnitude_fc4_prunedclass/ReadVariableOp2l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_14prune_low_magnitude_fc4_prunedclass/ReadVariableOp_12l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_24prune_low_magnitude_fc4_prunedclass/ReadVariableOp_22l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_34prune_low_magnitude_fc4_prunedclass/ReadVariableOp_32l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_44prune_low_magnitude_fc4_prunedclass/ReadVariableOp_42l
4prune_low_magnitude_fc4_prunedclass/ReadVariableOp_54prune_low_magnitude_fc4_prunedclass/ReadVariableOp_52�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�;
�
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1595160

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:'
readvariableop_3_resource:
identity��AssignVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5'
	no_updateNoOp*
_output_shapes
 )
no_update_1NoOp*
_output_shapes
 n
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype0r
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: }
ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A_
mul_1MulReadVariableOp:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
ReadVariableOp_1ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:
ReadVariableOp_2ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_5MulReadVariableOp_3:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�c
�
.prune_low_magnitude_fc3_prun_cond_true_1598019P
Fprune_low_magnitude_fc3_prun_cond_greaterequal_readvariableop_resource:	 [
Iprune_low_magnitude_fc3_prun_cond_pruning_ops_abs_readvariableop_resource:M
;prune_low_magnitude_fc3_prun_cond_assignvariableop_resource:G
=prune_low_magnitude_fc3_prun_cond_assignvariableop_1_resource: X
Tprune_low_magnitude_fc3_prun_cond_identity_prune_low_magnitude_fc3_prun_logicaland_1
0
,prune_low_magnitude_fc3_prun_cond_identity_1
��2prune_low_magnitude_fc3_prun/cond/AssignVariableOp�4prune_low_magnitude_fc3_prun/cond/AssignVariableOp_1�=prune_low_magnitude_fc3_prun/cond/GreaterEqual/ReadVariableOp�:prune_low_magnitude_fc3_prun/cond/LessEqual/ReadVariableOp�4prune_low_magnitude_fc3_prun/cond/Sub/ReadVariableOp�@prune_low_magnitude_fc3_prun/cond/pruning_ops/Abs/ReadVariableOp�
=prune_low_magnitude_fc3_prun/cond/GreaterEqual/ReadVariableOpReadVariableOpFprune_low_magnitude_fc3_prun_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	s
0prune_low_magnitude_fc3_prun/cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
.prune_low_magnitude_fc3_prun/cond/GreaterEqualGreaterEqualEprune_low_magnitude_fc3_prun/cond/GreaterEqual/ReadVariableOp:value:09prune_low_magnitude_fc3_prun/cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
:prune_low_magnitude_fc3_prun/cond/LessEqual/ReadVariableOpReadVariableOpFprune_low_magnitude_fc3_prun_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	p
-prune_low_magnitude_fc3_prun/cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N�
+prune_low_magnitude_fc3_prun/cond/LessEqual	LessEqualBprune_low_magnitude_fc3_prun/cond/LessEqual/ReadVariableOp:value:06prune_low_magnitude_fc3_prun/cond/LessEqual/y:output:0*
T0	*
_output_shapes
: k
(prune_low_magnitude_fc3_prun/cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�Nj
(prune_low_magnitude_fc3_prun/cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : �
&prune_low_magnitude_fc3_prun/cond/LessLess1prune_low_magnitude_fc3_prun/cond/Less/x:output:01prune_low_magnitude_fc3_prun/cond/Less/y:output:0*
T0*
_output_shapes
: �
+prune_low_magnitude_fc3_prun/cond/LogicalOr	LogicalOr/prune_low_magnitude_fc3_prun/cond/LessEqual:z:0*prune_low_magnitude_fc3_prun/cond/Less:z:0*
_output_shapes
: �
,prune_low_magnitude_fc3_prun/cond/LogicalAnd
LogicalAnd2prune_low_magnitude_fc3_prun/cond/GreaterEqual:z:0/prune_low_magnitude_fc3_prun/cond/LogicalOr:z:0*
_output_shapes
: �
4prune_low_magnitude_fc3_prun/cond/Sub/ReadVariableOpReadVariableOpFprune_low_magnitude_fc3_prun_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	j
'prune_low_magnitude_fc3_prun/cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
%prune_low_magnitude_fc3_prun/cond/SubSub<prune_low_magnitude_fc3_prun/cond/Sub/ReadVariableOp:value:00prune_low_magnitude_fc3_prun/cond/Sub/y:output:0*
T0	*
_output_shapes
: n
,prune_low_magnitude_fc3_prun/cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd�
*prune_low_magnitude_fc3_prun/cond/FloorModFloorMod)prune_low_magnitude_fc3_prun/cond/Sub:z:05prune_low_magnitude_fc3_prun/cond/FloorMod/y:output:0*
T0	*
_output_shapes
: k
)prune_low_magnitude_fc3_prun/cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'prune_low_magnitude_fc3_prun/cond/EqualEqual.prune_low_magnitude_fc3_prun/cond/FloorMod:z:02prune_low_magnitude_fc3_prun/cond/Equal/y:output:0*
T0	*
_output_shapes
: �
.prune_low_magnitude_fc3_prun/cond/LogicalAnd_1
LogicalAnd0prune_low_magnitude_fc3_prun/cond/LogicalAnd:z:0+prune_low_magnitude_fc3_prun/cond/Equal:z:0*
_output_shapes
: l
'prune_low_magnitude_fc3_prun/cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
@prune_low_magnitude_fc3_prun/cond/pruning_ops/Abs/ReadVariableOpReadVariableOpIprune_low_magnitude_fc3_prun_cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0�
1prune_low_magnitude_fc3_prun/cond/pruning_ops/AbsAbsHprune_low_magnitude_fc3_prun/cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:u
2prune_low_magnitude_fc3_prun/cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :��
2prune_low_magnitude_fc3_prun/cond/pruning_ops/CastCast;prune_low_magnitude_fc3_prun/cond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: x
3prune_low_magnitude_fc3_prun/cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
1prune_low_magnitude_fc3_prun/cond/pruning_ops/subSub<prune_low_magnitude_fc3_prun/cond/pruning_ops/sub/x:output:00prune_low_magnitude_fc3_prun/cond/Const:output:0*
T0*
_output_shapes
: �
1prune_low_magnitude_fc3_prun/cond/pruning_ops/mulMul6prune_low_magnitude_fc3_prun/cond/pruning_ops/Cast:y:05prune_low_magnitude_fc3_prun/cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: �
3prune_low_magnitude_fc3_prun/cond/pruning_ops/RoundRound5prune_low_magnitude_fc3_prun/cond/pruning_ops/mul:z:0*
T0*
_output_shapes
: |
7prune_low_magnitude_fc3_prun/cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
5prune_low_magnitude_fc3_prun/cond/pruning_ops/MaximumMaximum7prune_low_magnitude_fc3_prun/cond/pruning_ops/Round:y:0@prune_low_magnitude_fc3_prun/cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: �
4prune_low_magnitude_fc3_prun/cond/pruning_ops/Cast_1Cast9prune_low_magnitude_fc3_prun/cond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: �
;prune_low_magnitude_fc3_prun/cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
5prune_low_magnitude_fc3_prun/cond/pruning_ops/ReshapeReshape5prune_low_magnitude_fc3_prun/cond/pruning_ops/Abs:y:0Dprune_low_magnitude_fc3_prun/cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:�w
4prune_low_magnitude_fc3_prun/cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :��
4prune_low_magnitude_fc3_prun/cond/pruning_ops/TopKV2TopKV2>prune_low_magnitude_fc3_prun/cond/pruning_ops/Reshape:output:0=prune_low_magnitude_fc3_prun/cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:�:�w
5prune_low_magnitude_fc3_prun/cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
3prune_low_magnitude_fc3_prun/cond/pruning_ops/sub_1Sub8prune_low_magnitude_fc3_prun/cond/pruning_ops/Cast_1:y:0>prune_low_magnitude_fc3_prun/cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: }
;prune_low_magnitude_fc3_prun/cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6prune_low_magnitude_fc3_prun/cond/pruning_ops/GatherV2GatherV2=prune_low_magnitude_fc3_prun/cond/pruning_ops/TopKV2:values:07prune_low_magnitude_fc3_prun/cond/pruning_ops/sub_1:z:0Dprune_low_magnitude_fc3_prun/cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: w
5prune_low_magnitude_fc3_prun/cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
3prune_low_magnitude_fc3_prun/cond/pruning_ops/sub_2Sub8prune_low_magnitude_fc3_prun/cond/pruning_ops/Cast_1:y:0>prune_low_magnitude_fc3_prun/cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: 
=prune_low_magnitude_fc3_prun/cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8prune_low_magnitude_fc3_prun/cond/pruning_ops/GatherV2_1GatherV2>prune_low_magnitude_fc3_prun/cond/pruning_ops/TopKV2:indices:07prune_low_magnitude_fc3_prun/cond/pruning_ops/sub_2:z:0Fprune_low_magnitude_fc3_prun/cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
:prune_low_magnitude_fc3_prun/cond/pruning_ops/GreaterEqualGreaterEqual5prune_low_magnitude_fc3_prun/cond/pruning_ops/Abs:y:0?prune_low_magnitude_fc3_prun/cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:w
4prune_low_magnitude_fc3_prun/cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value
B :�}
;prune_low_magnitude_fc3_prun/cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z
=prune_low_magnitude_fc3_prun/cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
5prune_low_magnitude_fc3_prun/cond/pruning_ops/one_hotOneHotAprune_low_magnitude_fc3_prun/cond/pruning_ops/GatherV2_1:output:0=prune_low_magnitude_fc3_prun/cond/pruning_ops/Size_2:output:0Dprune_low_magnitude_fc3_prun/cond/pruning_ops/one_hot/Const:output:0Fprune_low_magnitude_fc3_prun/cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes	
:��
=prune_low_magnitude_fc3_prun/cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
7prune_low_magnitude_fc3_prun/cond/pruning_ops/Reshape_1Reshape>prune_low_magnitude_fc3_prun/cond/pruning_ops/one_hot:output:0Fprune_low_magnitude_fc3_prun/cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
7prune_low_magnitude_fc3_prun/cond/pruning_ops/LogicalOr	LogicalOr>prune_low_magnitude_fc3_prun/cond/pruning_ops/GreaterEqual:z:0@prune_low_magnitude_fc3_prun/cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:�
&prune_low_magnitude_fc3_prun/cond/CastCast;prune_low_magnitude_fc3_prun/cond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
2prune_low_magnitude_fc3_prun/cond/AssignVariableOpAssignVariableOp;prune_low_magnitude_fc3_prun_cond_assignvariableop_resource*prune_low_magnitude_fc3_prun/cond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
4prune_low_magnitude_fc3_prun/cond/AssignVariableOp_1AssignVariableOp=prune_low_magnitude_fc3_prun_cond_assignvariableop_1_resource?prune_low_magnitude_fc3_prun/cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
,prune_low_magnitude_fc3_prun/cond/group_depsNoOp3^prune_low_magnitude_fc3_prun/cond/AssignVariableOp5^prune_low_magnitude_fc3_prun/cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
*prune_low_magnitude_fc3_prun/cond/IdentityIdentityTprune_low_magnitude_fc3_prun_cond_identity_prune_low_magnitude_fc3_prun_logicaland_1-^prune_low_magnitude_fc3_prun/cond/group_deps*
T0
*
_output_shapes
: �
,prune_low_magnitude_fc3_prun/cond/Identity_1Identity3prune_low_magnitude_fc3_prun/cond/Identity:output:0'^prune_low_magnitude_fc3_prun/cond/NoOp*
T0
*
_output_shapes
: �
&prune_low_magnitude_fc3_prun/cond/NoOpNoOp3^prune_low_magnitude_fc3_prun/cond/AssignVariableOp5^prune_low_magnitude_fc3_prun/cond/AssignVariableOp_1>^prune_low_magnitude_fc3_prun/cond/GreaterEqual/ReadVariableOp;^prune_low_magnitude_fc3_prun/cond/LessEqual/ReadVariableOp5^prune_low_magnitude_fc3_prun/cond/Sub/ReadVariableOpA^prune_low_magnitude_fc3_prun/cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "e
,prune_low_magnitude_fc3_prun_cond_identity_15prune_low_magnitude_fc3_prun/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2h
2prune_low_magnitude_fc3_prun/cond/AssignVariableOp2prune_low_magnitude_fc3_prun/cond/AssignVariableOp2l
4prune_low_magnitude_fc3_prun/cond/AssignVariableOp_14prune_low_magnitude_fc3_prun/cond/AssignVariableOp_12~
=prune_low_magnitude_fc3_prun/cond/GreaterEqual/ReadVariableOp=prune_low_magnitude_fc3_prun/cond/GreaterEqual/ReadVariableOp2x
:prune_low_magnitude_fc3_prun/cond/LessEqual/ReadVariableOp:prune_low_magnitude_fc3_prun/cond/LessEqual/ReadVariableOp2l
4prune_low_magnitude_fc3_prun/cond/Sub/ReadVariableOp4prune_low_magnitude_fc3_prun/cond/Sub/ReadVariableOp2�
@prune_low_magnitude_fc3_prun/cond/pruning_ops/Abs/ReadVariableOp@prune_low_magnitude_fc3_prun/cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�
�
%__inference_signature_wrapper_1596849
encoder_input
unknown:@
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:(

unknown_13:(

unknown_14:(


unknown_15:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1594825o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������@: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�z
�
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1595907

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: '
readvariableop_4_resource:
identity��AssignVariableOp�AssignVariableOp_1�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: �
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(7
updateNoOp^AssignVariableOp*
_output_shapes
 �
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	X
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: [
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: �
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: �
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *H
else_branch9R7
5assert_greater_equal_Assert_AssertGuard_false_1595719*
output_shapes
: *G
then_branch8R6
4assert_greater_equal_Assert_AssertGuard_true_1595718�
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�{
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�No
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: |
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N{
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : O
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: G
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: Q

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: �
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	{
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�W
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RdS
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: |
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R O
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: M
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: }
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��L?�
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*%
else_branchR
cond_false_1595759*
output_shapes
: *$
then_branchR
cond_true_1595758q
cond/IdentityIdentitycond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: i
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 _
Mul/ReadVariableOpReadVariableOpcond_input_1*
_output_shapes

:*
dtype0h
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: r
ReadVariableOp_1ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ba
mul_1MulReadVariableOp_1:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_1AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_1:z:0*
T0*
_output_shapes

:[
add_2AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value/MinimumMinimum	add_2:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:r
ReadVariableOp_2ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_2:value:0*
T0*
_output_shapes

:M
add_3AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:r
ReadVariableOp_3ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0j
add_4AddV2ReadVariableOp_3:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_4:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_5MulReadVariableOp_4:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_5AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_5:z:0*
T0*
_output_shapes
:[
add_6AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_5:value:0*
T0*
_output_shapes
:I
add_7AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_7:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0f
add_8AddV2ReadVariableOp_6:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_8:z:0*
T0*'
_output_shapes
:����������
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^condJ^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond2�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�r
�
 __inference__traced_save_1600337
file_prefix)
%savev2_fc1_kernel_read_readvariableop'
#savev2_fc1_bias_read_readvariableop@
<savev2_prune_low_magnitude_fc2_prun_mask_read_readvariableopE
Asavev2_prune_low_magnitude_fc2_prun_threshold_read_readvariableopH
Dsavev2_prune_low_magnitude_fc2_prun_pruning_step_read_readvariableop	@
<savev2_prune_low_magnitude_fc3_prun_mask_read_readvariableopE
Asavev2_prune_low_magnitude_fc3_prun_threshold_read_readvariableopH
Dsavev2_prune_low_magnitude_fc3_prun_pruning_step_read_readvariableop	4
0savev2_encoder_output_kernel_read_readvariableop2
.savev2_encoder_output_bias_read_readvariableopG
Csavev2_prune_low_magnitude_fc4_prunedclass_mask_read_readvariableopL
Hsavev2_prune_low_magnitude_fc4_prunedclass_threshold_read_readvariableopO
Ksavev2_prune_low_magnitude_fc4_prunedclass_pruning_step_read_readvariableop	/
+savev2_fc5_class_kernel_read_readvariableop-
)savev2_fc5_class_bias_read_readvariableop4
0savev2_classifier_out_kernel_read_readvariableop2
.savev2_classifier_out_bias_read_readvariableopB
>savev2_prune_low_magnitude_fc2_prun_kernel_read_readvariableop@
<savev2_prune_low_magnitude_fc2_prun_bias_read_readvariableopB
>savev2_prune_low_magnitude_fc3_prun_kernel_read_readvariableop@
<savev2_prune_low_magnitude_fc3_prun_bias_read_readvariableopI
Esavev2_prune_low_magnitude_fc4_prunedclass_kernel_read_readvariableopG
Csavev2_prune_low_magnitude_fc4_prunedclass_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop0
,savev2_adam_m_fc1_kernel_read_readvariableop0
,savev2_adam_v_fc1_kernel_read_readvariableop.
*savev2_adam_m_fc1_bias_read_readvariableop.
*savev2_adam_v_fc1_bias_read_readvariableopI
Esavev2_adam_m_prune_low_magnitude_fc2_prun_kernel_read_readvariableopI
Esavev2_adam_v_prune_low_magnitude_fc2_prun_kernel_read_readvariableopG
Csavev2_adam_m_prune_low_magnitude_fc2_prun_bias_read_readvariableopG
Csavev2_adam_v_prune_low_magnitude_fc2_prun_bias_read_readvariableopI
Esavev2_adam_m_prune_low_magnitude_fc3_prun_kernel_read_readvariableopI
Esavev2_adam_v_prune_low_magnitude_fc3_prun_kernel_read_readvariableopG
Csavev2_adam_m_prune_low_magnitude_fc3_prun_bias_read_readvariableopG
Csavev2_adam_v_prune_low_magnitude_fc3_prun_bias_read_readvariableop;
7savev2_adam_m_encoder_output_kernel_read_readvariableop;
7savev2_adam_v_encoder_output_kernel_read_readvariableop9
5savev2_adam_m_encoder_output_bias_read_readvariableop9
5savev2_adam_v_encoder_output_bias_read_readvariableopP
Lsavev2_adam_m_prune_low_magnitude_fc4_prunedclass_kernel_read_readvariableopP
Lsavev2_adam_v_prune_low_magnitude_fc4_prunedclass_kernel_read_readvariableopN
Jsavev2_adam_m_prune_low_magnitude_fc4_prunedclass_bias_read_readvariableopN
Jsavev2_adam_v_prune_low_magnitude_fc4_prunedclass_bias_read_readvariableop6
2savev2_adam_m_fc5_class_kernel_read_readvariableop6
2savev2_adam_v_fc5_class_kernel_read_readvariableop4
0savev2_adam_m_fc5_class_bias_read_readvariableop4
0savev2_adam_v_fc5_class_bias_read_readvariableop;
7savev2_adam_m_classifier_out_kernel_read_readvariableop;
7savev2_adam_v_classifier_out_kernel_read_readvariableop9
5savev2_adam_m_classifier_out_bias_read_readvariableop9
5savev2_adam_v_classifier_out_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-2/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-4/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop<savev2_prune_low_magnitude_fc2_prun_mask_read_readvariableopAsavev2_prune_low_magnitude_fc2_prun_threshold_read_readvariableopDsavev2_prune_low_magnitude_fc2_prun_pruning_step_read_readvariableop<savev2_prune_low_magnitude_fc3_prun_mask_read_readvariableopAsavev2_prune_low_magnitude_fc3_prun_threshold_read_readvariableopDsavev2_prune_low_magnitude_fc3_prun_pruning_step_read_readvariableop0savev2_encoder_output_kernel_read_readvariableop.savev2_encoder_output_bias_read_readvariableopCsavev2_prune_low_magnitude_fc4_prunedclass_mask_read_readvariableopHsavev2_prune_low_magnitude_fc4_prunedclass_threshold_read_readvariableopKsavev2_prune_low_magnitude_fc4_prunedclass_pruning_step_read_readvariableop+savev2_fc5_class_kernel_read_readvariableop)savev2_fc5_class_bias_read_readvariableop0savev2_classifier_out_kernel_read_readvariableop.savev2_classifier_out_bias_read_readvariableop>savev2_prune_low_magnitude_fc2_prun_kernel_read_readvariableop<savev2_prune_low_magnitude_fc2_prun_bias_read_readvariableop>savev2_prune_low_magnitude_fc3_prun_kernel_read_readvariableop<savev2_prune_low_magnitude_fc3_prun_bias_read_readvariableopEsavev2_prune_low_magnitude_fc4_prunedclass_kernel_read_readvariableopCsavev2_prune_low_magnitude_fc4_prunedclass_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop,savev2_adam_m_fc1_kernel_read_readvariableop,savev2_adam_v_fc1_kernel_read_readvariableop*savev2_adam_m_fc1_bias_read_readvariableop*savev2_adam_v_fc1_bias_read_readvariableopEsavev2_adam_m_prune_low_magnitude_fc2_prun_kernel_read_readvariableopEsavev2_adam_v_prune_low_magnitude_fc2_prun_kernel_read_readvariableopCsavev2_adam_m_prune_low_magnitude_fc2_prun_bias_read_readvariableopCsavev2_adam_v_prune_low_magnitude_fc2_prun_bias_read_readvariableopEsavev2_adam_m_prune_low_magnitude_fc3_prun_kernel_read_readvariableopEsavev2_adam_v_prune_low_magnitude_fc3_prun_kernel_read_readvariableopCsavev2_adam_m_prune_low_magnitude_fc3_prun_bias_read_readvariableopCsavev2_adam_v_prune_low_magnitude_fc3_prun_bias_read_readvariableop7savev2_adam_m_encoder_output_kernel_read_readvariableop7savev2_adam_v_encoder_output_kernel_read_readvariableop5savev2_adam_m_encoder_output_bias_read_readvariableop5savev2_adam_v_encoder_output_bias_read_readvariableopLsavev2_adam_m_prune_low_magnitude_fc4_prunedclass_kernel_read_readvariableopLsavev2_adam_v_prune_low_magnitude_fc4_prunedclass_kernel_read_readvariableopJsavev2_adam_m_prune_low_magnitude_fc4_prunedclass_bias_read_readvariableopJsavev2_adam_v_prune_low_magnitude_fc4_prunedclass_bias_read_readvariableop2savev2_adam_m_fc5_class_kernel_read_readvariableop2savev2_adam_v_fc5_class_kernel_read_readvariableop0savev2_adam_m_fc5_class_bias_read_readvariableop0savev2_adam_v_fc5_class_bias_read_readvariableop7savev2_adam_m_classifier_out_kernel_read_readvariableop7savev2_adam_v_classifier_out_kernel_read_readvariableop5savev2_adam_m_classifier_out_bias_read_readvariableop5savev2_adam_v_classifier_out_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *H
dtypes>
<2:				�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :@::: : :: : :::: : :(:(:(
:
::::::: : :@:@:::::::::::::::::::(:(:(:(:(
:(
:
:
: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:(: 

_output_shapes
:(:$ 

_output_shapes

:(
: 

_output_shapes
:
:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::  

_output_shapes
:: !

_output_shapes
::$" 

_output_shapes

::$# 

_output_shapes

:: $

_output_shapes
:: %

_output_shapes
::$& 

_output_shapes

::$' 

_output_shapes

:: (

_output_shapes
:: )

_output_shapes
::$* 

_output_shapes

::$+ 

_output_shapes

:: ,

_output_shapes
:: -

_output_shapes
::$. 

_output_shapes

:(:$/ 

_output_shapes

:(: 0

_output_shapes
:(: 1

_output_shapes
:(:$2 

_output_shapes

:(
:$3 

_output_shapes

:(
: 4

_output_shapes
:
: 5

_output_shapes
:
:6
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
: 
�c
�
.prune_low_magnitude_fc2_prun_cond_true_1597776P
Fprune_low_magnitude_fc2_prun_cond_greaterequal_readvariableop_resource:	 [
Iprune_low_magnitude_fc2_prun_cond_pruning_ops_abs_readvariableop_resource:M
;prune_low_magnitude_fc2_prun_cond_assignvariableop_resource:G
=prune_low_magnitude_fc2_prun_cond_assignvariableop_1_resource: X
Tprune_low_magnitude_fc2_prun_cond_identity_prune_low_magnitude_fc2_prun_logicaland_1
0
,prune_low_magnitude_fc2_prun_cond_identity_1
��2prune_low_magnitude_fc2_prun/cond/AssignVariableOp�4prune_low_magnitude_fc2_prun/cond/AssignVariableOp_1�=prune_low_magnitude_fc2_prun/cond/GreaterEqual/ReadVariableOp�:prune_low_magnitude_fc2_prun/cond/LessEqual/ReadVariableOp�4prune_low_magnitude_fc2_prun/cond/Sub/ReadVariableOp�@prune_low_magnitude_fc2_prun/cond/pruning_ops/Abs/ReadVariableOp�
=prune_low_magnitude_fc2_prun/cond/GreaterEqual/ReadVariableOpReadVariableOpFprune_low_magnitude_fc2_prun_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	s
0prune_low_magnitude_fc2_prun/cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
.prune_low_magnitude_fc2_prun/cond/GreaterEqualGreaterEqualEprune_low_magnitude_fc2_prun/cond/GreaterEqual/ReadVariableOp:value:09prune_low_magnitude_fc2_prun/cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
:prune_low_magnitude_fc2_prun/cond/LessEqual/ReadVariableOpReadVariableOpFprune_low_magnitude_fc2_prun_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	p
-prune_low_magnitude_fc2_prun/cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N�
+prune_low_magnitude_fc2_prun/cond/LessEqual	LessEqualBprune_low_magnitude_fc2_prun/cond/LessEqual/ReadVariableOp:value:06prune_low_magnitude_fc2_prun/cond/LessEqual/y:output:0*
T0	*
_output_shapes
: k
(prune_low_magnitude_fc2_prun/cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�Nj
(prune_low_magnitude_fc2_prun/cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : �
&prune_low_magnitude_fc2_prun/cond/LessLess1prune_low_magnitude_fc2_prun/cond/Less/x:output:01prune_low_magnitude_fc2_prun/cond/Less/y:output:0*
T0*
_output_shapes
: �
+prune_low_magnitude_fc2_prun/cond/LogicalOr	LogicalOr/prune_low_magnitude_fc2_prun/cond/LessEqual:z:0*prune_low_magnitude_fc2_prun/cond/Less:z:0*
_output_shapes
: �
,prune_low_magnitude_fc2_prun/cond/LogicalAnd
LogicalAnd2prune_low_magnitude_fc2_prun/cond/GreaterEqual:z:0/prune_low_magnitude_fc2_prun/cond/LogicalOr:z:0*
_output_shapes
: �
4prune_low_magnitude_fc2_prun/cond/Sub/ReadVariableOpReadVariableOpFprune_low_magnitude_fc2_prun_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	j
'prune_low_magnitude_fc2_prun/cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
%prune_low_magnitude_fc2_prun/cond/SubSub<prune_low_magnitude_fc2_prun/cond/Sub/ReadVariableOp:value:00prune_low_magnitude_fc2_prun/cond/Sub/y:output:0*
T0	*
_output_shapes
: n
,prune_low_magnitude_fc2_prun/cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd�
*prune_low_magnitude_fc2_prun/cond/FloorModFloorMod)prune_low_magnitude_fc2_prun/cond/Sub:z:05prune_low_magnitude_fc2_prun/cond/FloorMod/y:output:0*
T0	*
_output_shapes
: k
)prune_low_magnitude_fc2_prun/cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'prune_low_magnitude_fc2_prun/cond/EqualEqual.prune_low_magnitude_fc2_prun/cond/FloorMod:z:02prune_low_magnitude_fc2_prun/cond/Equal/y:output:0*
T0	*
_output_shapes
: �
.prune_low_magnitude_fc2_prun/cond/LogicalAnd_1
LogicalAnd0prune_low_magnitude_fc2_prun/cond/LogicalAnd:z:0+prune_low_magnitude_fc2_prun/cond/Equal:z:0*
_output_shapes
: l
'prune_low_magnitude_fc2_prun/cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
@prune_low_magnitude_fc2_prun/cond/pruning_ops/Abs/ReadVariableOpReadVariableOpIprune_low_magnitude_fc2_prun_cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0�
1prune_low_magnitude_fc2_prun/cond/pruning_ops/AbsAbsHprune_low_magnitude_fc2_prun/cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:u
2prune_low_magnitude_fc2_prun/cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :��
2prune_low_magnitude_fc2_prun/cond/pruning_ops/CastCast;prune_low_magnitude_fc2_prun/cond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: x
3prune_low_magnitude_fc2_prun/cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
1prune_low_magnitude_fc2_prun/cond/pruning_ops/subSub<prune_low_magnitude_fc2_prun/cond/pruning_ops/sub/x:output:00prune_low_magnitude_fc2_prun/cond/Const:output:0*
T0*
_output_shapes
: �
1prune_low_magnitude_fc2_prun/cond/pruning_ops/mulMul6prune_low_magnitude_fc2_prun/cond/pruning_ops/Cast:y:05prune_low_magnitude_fc2_prun/cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: �
3prune_low_magnitude_fc2_prun/cond/pruning_ops/RoundRound5prune_low_magnitude_fc2_prun/cond/pruning_ops/mul:z:0*
T0*
_output_shapes
: |
7prune_low_magnitude_fc2_prun/cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
5prune_low_magnitude_fc2_prun/cond/pruning_ops/MaximumMaximum7prune_low_magnitude_fc2_prun/cond/pruning_ops/Round:y:0@prune_low_magnitude_fc2_prun/cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: �
4prune_low_magnitude_fc2_prun/cond/pruning_ops/Cast_1Cast9prune_low_magnitude_fc2_prun/cond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: �
;prune_low_magnitude_fc2_prun/cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
5prune_low_magnitude_fc2_prun/cond/pruning_ops/ReshapeReshape5prune_low_magnitude_fc2_prun/cond/pruning_ops/Abs:y:0Dprune_low_magnitude_fc2_prun/cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:�w
4prune_low_magnitude_fc2_prun/cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :��
4prune_low_magnitude_fc2_prun/cond/pruning_ops/TopKV2TopKV2>prune_low_magnitude_fc2_prun/cond/pruning_ops/Reshape:output:0=prune_low_magnitude_fc2_prun/cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:�:�w
5prune_low_magnitude_fc2_prun/cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
3prune_low_magnitude_fc2_prun/cond/pruning_ops/sub_1Sub8prune_low_magnitude_fc2_prun/cond/pruning_ops/Cast_1:y:0>prune_low_magnitude_fc2_prun/cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: }
;prune_low_magnitude_fc2_prun/cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6prune_low_magnitude_fc2_prun/cond/pruning_ops/GatherV2GatherV2=prune_low_magnitude_fc2_prun/cond/pruning_ops/TopKV2:values:07prune_low_magnitude_fc2_prun/cond/pruning_ops/sub_1:z:0Dprune_low_magnitude_fc2_prun/cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: w
5prune_low_magnitude_fc2_prun/cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
3prune_low_magnitude_fc2_prun/cond/pruning_ops/sub_2Sub8prune_low_magnitude_fc2_prun/cond/pruning_ops/Cast_1:y:0>prune_low_magnitude_fc2_prun/cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: 
=prune_low_magnitude_fc2_prun/cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8prune_low_magnitude_fc2_prun/cond/pruning_ops/GatherV2_1GatherV2>prune_low_magnitude_fc2_prun/cond/pruning_ops/TopKV2:indices:07prune_low_magnitude_fc2_prun/cond/pruning_ops/sub_2:z:0Fprune_low_magnitude_fc2_prun/cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
:prune_low_magnitude_fc2_prun/cond/pruning_ops/GreaterEqualGreaterEqual5prune_low_magnitude_fc2_prun/cond/pruning_ops/Abs:y:0?prune_low_magnitude_fc2_prun/cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:w
4prune_low_magnitude_fc2_prun/cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value
B :�}
;prune_low_magnitude_fc2_prun/cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z
=prune_low_magnitude_fc2_prun/cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
5prune_low_magnitude_fc2_prun/cond/pruning_ops/one_hotOneHotAprune_low_magnitude_fc2_prun/cond/pruning_ops/GatherV2_1:output:0=prune_low_magnitude_fc2_prun/cond/pruning_ops/Size_2:output:0Dprune_low_magnitude_fc2_prun/cond/pruning_ops/one_hot/Const:output:0Fprune_low_magnitude_fc2_prun/cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes	
:��
=prune_low_magnitude_fc2_prun/cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
7prune_low_magnitude_fc2_prun/cond/pruning_ops/Reshape_1Reshape>prune_low_magnitude_fc2_prun/cond/pruning_ops/one_hot:output:0Fprune_low_magnitude_fc2_prun/cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
7prune_low_magnitude_fc2_prun/cond/pruning_ops/LogicalOr	LogicalOr>prune_low_magnitude_fc2_prun/cond/pruning_ops/GreaterEqual:z:0@prune_low_magnitude_fc2_prun/cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:�
&prune_low_magnitude_fc2_prun/cond/CastCast;prune_low_magnitude_fc2_prun/cond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
2prune_low_magnitude_fc2_prun/cond/AssignVariableOpAssignVariableOp;prune_low_magnitude_fc2_prun_cond_assignvariableop_resource*prune_low_magnitude_fc2_prun/cond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
4prune_low_magnitude_fc2_prun/cond/AssignVariableOp_1AssignVariableOp=prune_low_magnitude_fc2_prun_cond_assignvariableop_1_resource?prune_low_magnitude_fc2_prun/cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
,prune_low_magnitude_fc2_prun/cond/group_depsNoOp3^prune_low_magnitude_fc2_prun/cond/AssignVariableOp5^prune_low_magnitude_fc2_prun/cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
*prune_low_magnitude_fc2_prun/cond/IdentityIdentityTprune_low_magnitude_fc2_prun_cond_identity_prune_low_magnitude_fc2_prun_logicaland_1-^prune_low_magnitude_fc2_prun/cond/group_deps*
T0
*
_output_shapes
: �
,prune_low_magnitude_fc2_prun/cond/Identity_1Identity3prune_low_magnitude_fc2_prun/cond/Identity:output:0'^prune_low_magnitude_fc2_prun/cond/NoOp*
T0
*
_output_shapes
: �
&prune_low_magnitude_fc2_prun/cond/NoOpNoOp3^prune_low_magnitude_fc2_prun/cond/AssignVariableOp5^prune_low_magnitude_fc2_prun/cond/AssignVariableOp_1>^prune_low_magnitude_fc2_prun/cond/GreaterEqual/ReadVariableOp;^prune_low_magnitude_fc2_prun/cond/LessEqual/ReadVariableOp5^prune_low_magnitude_fc2_prun/cond/Sub/ReadVariableOpA^prune_low_magnitude_fc2_prun/cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "e
,prune_low_magnitude_fc2_prun_cond_identity_15prune_low_magnitude_fc2_prun/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2h
2prune_low_magnitude_fc2_prun/cond/AssignVariableOp2prune_low_magnitude_fc2_prun/cond/AssignVariableOp2l
4prune_low_magnitude_fc2_prun/cond/AssignVariableOp_14prune_low_magnitude_fc2_prun/cond/AssignVariableOp_12~
=prune_low_magnitude_fc2_prun/cond/GreaterEqual/ReadVariableOp=prune_low_magnitude_fc2_prun/cond/GreaterEqual/ReadVariableOp2x
:prune_low_magnitude_fc2_prun/cond/LessEqual/ReadVariableOp:prune_low_magnitude_fc2_prun/cond/LessEqual/ReadVariableOp2l
4prune_low_magnitude_fc2_prun/cond/Sub/ReadVariableOp4prune_low_magnitude_fc2_prun/cond/Sub/ReadVariableOp2�
@prune_low_magnitude_fc2_prun/cond/pruning_ops/Abs/ReadVariableOp@prune_low_magnitude_fc2_prun/cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�
�
4assert_greater_equal_Assert_AssertGuard_true_1595718M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
r
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�n
�
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1599080

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: '
readvariableop_4_resource:
identity��AssignVariableOp�AssignVariableOp_1�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: �
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(7
updateNoOp^AssignVariableOp*
_output_shapes
 �
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	X
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: [
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: �
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: �
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *H
else_branch9R7
5assert_greater_equal_Assert_AssertGuard_false_1598898*
output_shapes
: *G
then_branch8R6
4assert_greater_equal_Assert_AssertGuard_true_1598897�
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�{
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�No
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: |
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N{
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : O
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: G
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: Q

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: �
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	{
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�W
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RdS
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: |
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R O
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: M
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: }
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��L?�
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*%
else_branchR
cond_false_1598938*
output_shapes
: *$
then_branchR
cond_true_1598937q
cond/IdentityIdentitycond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: i
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 _
Mul/ReadVariableOpReadVariableOpcond_input_1*
_output_shapes

:*
dtype0h
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: r
ReadVariableOp_1ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aa
mul_1MulReadVariableOp_1:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_1AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_1:z:0*
T0*
_output_shapes

:[
add_2AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_2:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:r
ReadVariableOp_2ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_2:value:0*
T0*
_output_shapes

:M
add_3AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:r
ReadVariableOp_3ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0j
add_4AddV2ReadVariableOp_3:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_4:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_5MulReadVariableOp_4:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_5AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_5:z:0*
T0*
_output_shapes
:[
add_6AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_5:value:0*
T0*
_output_shapes
:I
add_7AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_7:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0f
add_8AddV2ReadVariableOp_6:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_8:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_prune_low_magnitude_fc2_prun_layer_call_fn_1598806

inputs
unknown:	 
	unknown_0:
	unknown_1:
	unknown_2: 
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1596387o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�n
�
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1596152

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: '
readvariableop_4_resource:
identity��AssignVariableOp�AssignVariableOp_1�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: �
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(7
updateNoOp^AssignVariableOp*
_output_shapes
 �
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	X
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: [
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: �
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: �
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *H
else_branch9R7
5assert_greater_equal_Assert_AssertGuard_false_1595970*
output_shapes
: *G
then_branch8R6
4assert_greater_equal_Assert_AssertGuard_true_1595969�
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�{
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�No
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: |
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N{
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : O
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: G
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: Q

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: �
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	{
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�W
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RdS
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: |
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R O
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: M
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: }
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��L?�
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*%
else_branchR
cond_false_1596010*
output_shapes
: *$
then_branchR
cond_true_1596009q
cond/IdentityIdentitycond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: i
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 _
Mul/ReadVariableOpReadVariableOpcond_input_1*
_output_shapes

:*
dtype0h
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: r
ReadVariableOp_1ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aa
mul_1MulReadVariableOp_1:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_1AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_1:z:0*
T0*
_output_shapes

:[
add_2AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_2:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:r
ReadVariableOp_2ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_2:value:0*
T0*
_output_shapes

:M
add_3AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:r
ReadVariableOp_3ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0j
add_4AddV2ReadVariableOp_3:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_4:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_5MulReadVariableOp_4:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_5AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_5:z:0*
T0*
_output_shapes
:[
add_6AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_5:value:0*
T0*
_output_shapes
:I
add_7AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_7:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0f
add_8AddV2ReadVariableOp_6:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_8:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_prune_low_magnitude_fc3_prun_layer_call_fn_1599160

inputs
unknown:	 
	unknown_0:
	unknown_1:
	unknown_2: 
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1596152o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�n
�
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1596387

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: '
readvariableop_4_resource:
identity��AssignVariableOp�AssignVariableOp_1�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2ReadVariableOp:value:0add/y:output:0*
T0	*
_output_shapes
: �
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(7
updateNoOp^AssignVariableOp*
_output_shapes
 �
#assert_greater_equal/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0	X
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: [
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: �
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: �
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *H
else_branch9R7
5assert_greater_equal_Assert_AssertGuard_false_1596205*
output_shapes
: *G
then_branch8R6
4assert_greater_equal_Assert_AssertGuard_true_1596204�
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
GreaterEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�{
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
LessEqual/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�No
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: |
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N{
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : O
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: G
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: Q

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: �
Sub/ReadVariableOpReadVariableOpreadvariableop_resource^AssignVariableOp1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	{
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�W
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RdS
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: |
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R O
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: M
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: }
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��L?�
condIfLogicalAnd_1:z:0readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0^AssignVariableOp*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*%
else_branchR
cond_false_1596245*
output_shapes
: *$
then_branchR
cond_true_1596244q
cond/IdentityIdentitycond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: i
update_1NoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 _
Mul/ReadVariableOpReadVariableOpcond_input_1*
_output_shapes

:*
dtype0h
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOp_1AssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: r
ReadVariableOp_1ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aa
mul_1MulReadVariableOp_1:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_1AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_1:z:0*
T0*
_output_shapes

:[
add_2AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_2:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:r
ReadVariableOp_2ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_2:value:0*
T0*
_output_shapes

:M
add_3AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:r
ReadVariableOp_3ReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0j
add_4AddV2ReadVariableOp_3:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_4:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_5MulReadVariableOp_4:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_5AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_5:z:0*
T0*
_output_shapes
:[
add_6AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_5:value:0*
T0*
_output_shapes
:I
add_7AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_7:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0f
add_8AddV2ReadVariableOp_6:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_8:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Xprune_low_magnitude_fc4_prunedclass_assert_greater_equal_Assert_AssertGuard_true_1598229�
�prune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc4_prunedclass_assert_greater_equal_all
[
Wprune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_placeholder	]
Yprune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_placeholder_1	Z
Vprune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_identity_1
�
Pprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Tprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc4_prunedclass_assert_greater_equal_allQ^prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Vprune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity_1Identity]prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "�
Vprune_low_magnitude_fc4_prunedclass_assert_greater_equal_assert_assertguard_identity_1_prune_low_magnitude_fc4_prunedclass/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�;
�
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1599232

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:'
readvariableop_3_resource:
identity��AssignVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5'
	no_updateNoOp*
_output_shapes
 )
no_update_1NoOp*
_output_shapes
 n
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype0r
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: }
ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A_
mul_1MulReadVariableOp:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
ReadVariableOp_1ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:
ReadVariableOp_2ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_5MulReadVariableOp_3:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
F__inference_fc5_class_layer_call_and_return_conditional_losses_1595447

inputs)
readvariableop_resource:('
readvariableop_3_resource:(
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�/fc5_class/kernel/Regularizer/Abs/ReadVariableOpG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:(N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:(@
NegNegtruediv:z:0*
T0*
_output_shapes

:(D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:(I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:(N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:([
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:(\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:(R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:(P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:(L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:(h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:(M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:(L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:(R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:(h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:(U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������(I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:(*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:(P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:(@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:(D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:(K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:(N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:([
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:(^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:(R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:(P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:(L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:(f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:(*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:(I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:(L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:(N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:(f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:(*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:(a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������(�
/fc5_class/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0�
 fc5_class/kernel/Regularizer/AbsAbs7fc5_class/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(s
"fc5_class/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 fc5_class/kernel/Regularizer/SumSum$fc5_class/kernel/Regularizer/Abs:y:0+fc5_class/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"fc5_class/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
 fc5_class/kernel/Regularizer/mulMul+fc5_class/kernel/Regularizer/mul/x:output:0)fc5_class/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_50^fc5_class/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52b
/fc5_class/kernel/Regularizer/Abs/ReadVariableOp/fc5_class/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
b
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1599488

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
C
'__inference_relu1_layer_call_fn_1598731

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_1594955`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_fc1_layer_call_fn_1598658

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_1594900o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
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
�
4assert_greater_equal_Assert_AssertGuard_true_1596204M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
r
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�[
�
G__inference_classifier_layer_call_and_return_conditional_losses_1596707
encoder_input
fc1_1596641:@
fc1_1596643:6
$prune_low_magnitude_fc2_prun_1596647:6
$prune_low_magnitude_fc2_prun_1596649:2
$prune_low_magnitude_fc2_prun_1596651:6
$prune_low_magnitude_fc3_prun_1596655:6
$prune_low_magnitude_fc3_prun_1596657:2
$prune_low_magnitude_fc3_prun_1596659:(
encoder_output_1596663:$
encoder_output_1596665:=
+prune_low_magnitude_fc4_prunedclass_1596668:=
+prune_low_magnitude_fc4_prunedclass_1596670:9
+prune_low_magnitude_fc4_prunedclass_1596672:#
fc5_class_1596676:(
fc5_class_1596678:((
classifier_out_1596682:(
$
classifier_out_1596684:

identity��&classifier_out/StatefulPartitionedCall�4classifier_out/kernel/Regularizer/Abs/ReadVariableOp�&encoder_output/StatefulPartitionedCall�fc1/StatefulPartitionedCall�!fc5_class/StatefulPartitionedCall�/fc5_class/kernel/Regularizer/Abs/ReadVariableOp�4prune_low_magnitude_fc2_prun/StatefulPartitionedCall�4prune_low_magnitude_fc3_prun/StatefulPartitionedCall�;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp�
fc1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputfc1_1596641fc1_1596643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_1594900�
relu1/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_1594955�
4prune_low_magnitude_fc2_prun/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0$prune_low_magnitude_fc2_prun_1596647$prune_low_magnitude_fc2_prun_1596649$prune_low_magnitude_fc2_prun_1596651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1595029�
relu2/PartitionedCallPartitionedCall=prune_low_magnitude_fc2_prun/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_1595086�
4prune_low_magnitude_fc3_prun/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0$prune_low_magnitude_fc3_prun_1596655$prune_low_magnitude_fc3_prun_1596657$prune_low_magnitude_fc3_prun_1596659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1595160�
relu3_enc/PartitionedCallPartitionedCall=prune_low_magnitude_fc3_prun/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1595217�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall"relu3_enc/PartitionedCall:output:0encoder_output_1596663encoder_output_1596665*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_output_layer_call_and_return_conditional_losses_1595230�
;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0+prune_low_magnitude_fc4_prunedclass_1596668+prune_low_magnitude_fc4_prunedclass_1596670+prune_low_magnitude_fc4_prunedclass_1596672*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *i
fdRb
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1595314�
prunclass_relu4/PartitionedCallPartitionedCallDprune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1595371�
!fc5_class/StatefulPartitionedCallStatefulPartitionedCall(prunclass_relu4/PartitionedCall:output:0fc5_class_1596676fc5_class_1596678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_fc5_class_layer_call_and_return_conditional_losses_1595447�
class_relu5/PartitionedCallPartitionedCall*fc5_class/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_class_relu5_layer_call_and_return_conditional_losses_1595502�
&classifier_out/StatefulPartitionedCallStatefulPartitionedCall$class_relu5/PartitionedCall:output:0classifier_out_1596682classifier_out_1596684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_classifier_out_layer_call_and_return_conditional_losses_1595578�
!classifier_output/PartitionedCallPartitionedCall/classifier_out/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_classifier_output_layer_call_and_return_conditional_losses_1595589�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+prune_low_magnitude_fc4_prunedclass_1596668<^prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
/fc5_class/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfc5_class_1596676*
_output_shapes

:(*
dtype0�
 fc5_class/kernel/Regularizer/AbsAbs7fc5_class/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(s
"fc5_class/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 fc5_class/kernel/Regularizer/SumSum$fc5_class/kernel/Regularizer/Abs:y:0+fc5_class/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"fc5_class/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
 fc5_class/kernel/Regularizer/mulMul+fc5_class/kernel/Regularizer/mul/x:output:0)fc5_class/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4classifier_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpclassifier_out_1596682*
_output_shapes

:(
*
dtype0�
%classifier_out/kernel/Regularizer/AbsAbs<classifier_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(
x
'classifier_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%classifier_out/kernel/Regularizer/SumSum)classifier_out/kernel/Regularizer/Abs:y:00classifier_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'classifier_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%classifier_out/kernel/Regularizer/mulMul0classifier_out/kernel/Regularizer/mul/x:output:0.classifier_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*classifier_output/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp'^classifier_out/StatefulPartitionedCall5^classifier_out/kernel/Regularizer/Abs/ReadVariableOp'^encoder_output/StatefulPartitionedCall^fc1/StatefulPartitionedCall"^fc5_class/StatefulPartitionedCall0^fc5_class/kernel/Regularizer/Abs/ReadVariableOp5^prune_low_magnitude_fc2_prun/StatefulPartitionedCall5^prune_low_magnitude_fc3_prun/StatefulPartitionedCall<^prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCallJ^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������@: : : : : : : : : : : : : : : : : 2P
&classifier_out/StatefulPartitionedCall&classifier_out/StatefulPartitionedCall2l
4classifier_out/kernel/Regularizer/Abs/ReadVariableOp4classifier_out/kernel/Regularizer/Abs/ReadVariableOp2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2F
!fc5_class/StatefulPartitionedCall!fc5_class/StatefulPartitionedCall2b
/fc5_class/kernel/Regularizer/Abs/ReadVariableOp/fc5_class/kernel/Regularizer/Abs/ReadVariableOp2l
4prune_low_magnitude_fc2_prun/StatefulPartitionedCall4prune_low_magnitude_fc2_prun/StatefulPartitionedCall2l
4prune_low_magnitude_fc3_prun/StatefulPartitionedCall4prune_low_magnitude_fc3_prun/StatefulPartitionedCall2z
;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall2�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
� 
^
B__inference_relu2_layer_call_and_return_conditional_losses_1595086

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_prune_low_magnitude_fc3_prun_layer_call_fn_1599145

inputs
unknown:
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1595160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4assert_greater_equal_Assert_AssertGuard_true_1595969M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
r
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�_
�
G__inference_classifier_layer_call_and_return_conditional_losses_1596538

inputs
fc1_1596460:@
fc1_1596462:.
$prune_low_magnitude_fc2_prun_1596466:	 6
$prune_low_magnitude_fc2_prun_1596468:6
$prune_low_magnitude_fc2_prun_1596470:.
$prune_low_magnitude_fc2_prun_1596472: 2
$prune_low_magnitude_fc2_prun_1596474:.
$prune_low_magnitude_fc3_prun_1596478:	 6
$prune_low_magnitude_fc3_prun_1596480:6
$prune_low_magnitude_fc3_prun_1596482:.
$prune_low_magnitude_fc3_prun_1596484: 2
$prune_low_magnitude_fc3_prun_1596486:(
encoder_output_1596490:$
encoder_output_1596492:5
+prune_low_magnitude_fc4_prunedclass_1596495:	 =
+prune_low_magnitude_fc4_prunedclass_1596497:=
+prune_low_magnitude_fc4_prunedclass_1596499:5
+prune_low_magnitude_fc4_prunedclass_1596501: 9
+prune_low_magnitude_fc4_prunedclass_1596503:#
fc5_class_1596507:(
fc5_class_1596509:((
classifier_out_1596513:(
$
classifier_out_1596515:

identity��&classifier_out/StatefulPartitionedCall�4classifier_out/kernel/Regularizer/Abs/ReadVariableOp�&encoder_output/StatefulPartitionedCall�fc1/StatefulPartitionedCall�!fc5_class/StatefulPartitionedCall�/fc5_class/kernel/Regularizer/Abs/ReadVariableOp�4prune_low_magnitude_fc2_prun/StatefulPartitionedCall�4prune_low_magnitude_fc3_prun/StatefulPartitionedCall�;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp�
fc1/StatefulPartitionedCallStatefulPartitionedCallinputsfc1_1596460fc1_1596462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_1594900�
relu1/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_1594955�
4prune_low_magnitude_fc2_prun/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0$prune_low_magnitude_fc2_prun_1596466$prune_low_magnitude_fc2_prun_1596468$prune_low_magnitude_fc2_prun_1596470$prune_low_magnitude_fc2_prun_1596472$prune_low_magnitude_fc2_prun_1596474*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1596387�
relu2/PartitionedCallPartitionedCall=prune_low_magnitude_fc2_prun/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_1595086�
4prune_low_magnitude_fc3_prun/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0$prune_low_magnitude_fc3_prun_1596478$prune_low_magnitude_fc3_prun_1596480$prune_low_magnitude_fc3_prun_1596482$prune_low_magnitude_fc3_prun_1596484$prune_low_magnitude_fc3_prun_1596486*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1596152�
relu3_enc/PartitionedCallPartitionedCall=prune_low_magnitude_fc3_prun/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1595217�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall"relu3_enc/PartitionedCall:output:0encoder_output_1596490encoder_output_1596492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_output_layer_call_and_return_conditional_losses_1595230�
;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0+prune_low_magnitude_fc4_prunedclass_1596495+prune_low_magnitude_fc4_prunedclass_1596497+prune_low_magnitude_fc4_prunedclass_1596499+prune_low_magnitude_fc4_prunedclass_1596501+prune_low_magnitude_fc4_prunedclass_1596503*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *i
fdRb
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1595907�
prunclass_relu4/PartitionedCallPartitionedCallDprune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1595371�
!fc5_class/StatefulPartitionedCallStatefulPartitionedCall(prunclass_relu4/PartitionedCall:output:0fc5_class_1596507fc5_class_1596509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_fc5_class_layer_call_and_return_conditional_losses_1595447�
class_relu5/PartitionedCallPartitionedCall*fc5_class/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_class_relu5_layer_call_and_return_conditional_losses_1595502�
&classifier_out/StatefulPartitionedCallStatefulPartitionedCall$class_relu5/PartitionedCall:output:0classifier_out_1596513classifier_out_1596515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_classifier_out_layer_call_and_return_conditional_losses_1595578�
!classifier_output/PartitionedCallPartitionedCall/classifier_out/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_classifier_output_layer_call_and_return_conditional_losses_1595589�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+prune_low_magnitude_fc4_prunedclass_1596497<^prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
/fc5_class/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpfc5_class_1596507*
_output_shapes

:(*
dtype0�
 fc5_class/kernel/Regularizer/AbsAbs7fc5_class/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(s
"fc5_class/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 fc5_class/kernel/Regularizer/SumSum$fc5_class/kernel/Regularizer/Abs:y:0+fc5_class/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"fc5_class/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
 fc5_class/kernel/Regularizer/mulMul+fc5_class/kernel/Regularizer/mul/x:output:0)fc5_class/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
4classifier_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpclassifier_out_1596513*
_output_shapes

:(
*
dtype0�
%classifier_out/kernel/Regularizer/AbsAbs<classifier_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(
x
'classifier_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%classifier_out/kernel/Regularizer/SumSum)classifier_out/kernel/Regularizer/Abs:y:00classifier_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'classifier_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%classifier_out/kernel/Regularizer/mulMul0classifier_out/kernel/Regularizer/mul/x:output:0.classifier_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*classifier_output/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp'^classifier_out/StatefulPartitionedCall5^classifier_out/kernel/Regularizer/Abs/ReadVariableOp'^encoder_output/StatefulPartitionedCall^fc1/StatefulPartitionedCall"^fc5_class/StatefulPartitionedCall0^fc5_class/kernel/Regularizer/Abs/ReadVariableOp5^prune_low_magnitude_fc2_prun/StatefulPartitionedCall5^prune_low_magnitude_fc3_prun/StatefulPartitionedCall<^prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCallJ^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������@: : : : : : : : : : : : : : : : : : : : : : : 2P
&classifier_out/StatefulPartitionedCall&classifier_out/StatefulPartitionedCall2l
4classifier_out/kernel/Regularizer/Abs/ReadVariableOp4classifier_out/kernel/Regularizer/Abs/ReadVariableOp2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2F
!fc5_class/StatefulPartitionedCall!fc5_class/StatefulPartitionedCall2b
/fc5_class/kernel/Regularizer/Abs/ReadVariableOp/fc5_class/kernel/Regularizer/Abs/ReadVariableOp2l
4prune_low_magnitude_fc2_prun/StatefulPartitionedCall4prune_low_magnitude_fc2_prun/StatefulPartitionedCall2l
4prune_low_magnitude_fc3_prun/StatefulPartitionedCall4prune_low_magnitude_fc3_prun/StatefulPartitionedCall2z
;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall;prune_low_magnitude_fc4_prunedclass/StatefulPartitionedCall2�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
"__inference__wrapped_model_1594825
encoder_input8
&classifier_fc1_readvariableop_resource:@6
(classifier_fc1_readvariableop_3_resource:U
Cclassifier_prune_low_magnitude_fc2_prun_mul_readvariableop_resource:W
Eclassifier_prune_low_magnitude_fc2_prun_mul_readvariableop_1_resource:O
Aclassifier_prune_low_magnitude_fc2_prun_readvariableop_3_resource:U
Cclassifier_prune_low_magnitude_fc3_prun_mul_readvariableop_resource:W
Eclassifier_prune_low_magnitude_fc3_prun_mul_readvariableop_1_resource:O
Aclassifier_prune_low_magnitude_fc3_prun_readvariableop_3_resource:J
8classifier_encoder_output_matmul_readvariableop_resource:G
9classifier_encoder_output_biasadd_readvariableop_resource:\
Jclassifier_prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource:^
Lclassifier_prune_low_magnitude_fc4_prunedclass_mul_readvariableop_1_resource:V
Hclassifier_prune_low_magnitude_fc4_prunedclass_readvariableop_3_resource:>
,classifier_fc5_class_readvariableop_resource:(<
.classifier_fc5_class_readvariableop_3_resource:(C
1classifier_classifier_out_readvariableop_resource:(
A
3classifier_classifier_out_readvariableop_3_resource:

identity��(classifier/classifier_out/ReadVariableOp�*classifier/classifier_out/ReadVariableOp_1�*classifier/classifier_out/ReadVariableOp_2�*classifier/classifier_out/ReadVariableOp_3�*classifier/classifier_out/ReadVariableOp_4�*classifier/classifier_out/ReadVariableOp_5�0classifier/encoder_output/BiasAdd/ReadVariableOp�/classifier/encoder_output/MatMul/ReadVariableOp�classifier/fc1/ReadVariableOp�classifier/fc1/ReadVariableOp_1�classifier/fc1/ReadVariableOp_2�classifier/fc1/ReadVariableOp_3�classifier/fc1/ReadVariableOp_4�classifier/fc1/ReadVariableOp_5�#classifier/fc5_class/ReadVariableOp�%classifier/fc5_class/ReadVariableOp_1�%classifier/fc5_class/ReadVariableOp_2�%classifier/fc5_class/ReadVariableOp_3�%classifier/fc5_class/ReadVariableOp_4�%classifier/fc5_class/ReadVariableOp_5�8classifier/prune_low_magnitude_fc2_prun/AssignVariableOp�:classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp�<classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1�6classifier/prune_low_magnitude_fc2_prun/ReadVariableOp�8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_1�8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_2�8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_3�8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_4�8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_5�8classifier/prune_low_magnitude_fc3_prun/AssignVariableOp�:classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp�<classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1�6classifier/prune_low_magnitude_fc3_prun/ReadVariableOp�8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_1�8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_2�8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_3�8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_4�8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_5�?classifier/prune_low_magnitude_fc4_prunedclass/AssignVariableOp�Aclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp�Cclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1�=classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp�?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1�?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2�?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3�?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4�?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5V
classifier/fc1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :V
classifier/fc1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : x
classifier/fc1/PowPowclassifier/fc1/Pow/x:output:0classifier/fc1/Pow/y:output:0*
T0*
_output_shapes
: c
classifier/fc1/CastCastclassifier/fc1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
classifier/fc1/ReadVariableOpReadVariableOp&classifier_fc1_readvariableop_resource*
_output_shapes

:@*
dtype0Y
classifier/fc1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
classifier/fc1/mulMul%classifier/fc1/ReadVariableOp:value:0classifier/fc1/mul/y:output:0*
T0*
_output_shapes

:@{
classifier/fc1/truedivRealDivclassifier/fc1/mul:z:0classifier/fc1/Cast:y:0*
T0*
_output_shapes

:@^
classifier/fc1/NegNegclassifier/fc1/truediv:z:0*
T0*
_output_shapes

:@b
classifier/fc1/RoundRoundclassifier/fc1/truediv:z:0*
T0*
_output_shapes

:@v
classifier/fc1/addAddV2classifier/fc1/Neg:y:0classifier/fc1/Round:y:0*
T0*
_output_shapes

:@l
classifier/fc1/StopGradientStopGradientclassifier/fc1/add:z:0*
T0*
_output_shapes

:@�
classifier/fc1/add_1AddV2classifier/fc1/truediv:z:0$classifier/fc1/StopGradient:output:0*
T0*
_output_shapes

:@k
&classifier/fc1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
$classifier/fc1/clip_by_value/MinimumMinimumclassifier/fc1/add_1:z:0/classifier/fc1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@c
classifier/fc1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
classifier/fc1/clip_by_valueMaximum(classifier/fc1/clip_by_value/Minimum:z:0'classifier/fc1/clip_by_value/y:output:0*
T0*
_output_shapes

:@
classifier/fc1/mul_1Mulclassifier/fc1/Cast:y:0 classifier/fc1/clip_by_value:z:0*
T0*
_output_shapes

:@_
classifier/fc1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
classifier/fc1/truediv_1RealDivclassifier/fc1/mul_1:z:0#classifier/fc1/truediv_1/y:output:0*
T0*
_output_shapes

:@[
classifier/fc1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/fc1/mul_2Mulclassifier/fc1/mul_2/x:output:0classifier/fc1/truediv_1:z:0*
T0*
_output_shapes

:@�
classifier/fc1/ReadVariableOp_1ReadVariableOp&classifier_fc1_readvariableop_resource*
_output_shapes

:@*
dtype0m
classifier/fc1/Neg_1Neg'classifier/fc1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:@z
classifier/fc1/add_2AddV2classifier/fc1/Neg_1:y:0classifier/fc1/mul_2:z:0*
T0*
_output_shapes

:@[
classifier/fc1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
classifier/fc1/mul_3Mulclassifier/fc1/mul_3/x:output:0classifier/fc1/add_2:z:0*
T0*
_output_shapes

:@p
classifier/fc1/StopGradient_1StopGradientclassifier/fc1/mul_3:z:0*
T0*
_output_shapes

:@�
classifier/fc1/ReadVariableOp_2ReadVariableOp&classifier_fc1_readvariableop_resource*
_output_shapes

:@*
dtype0�
classifier/fc1/add_3AddV2'classifier/fc1/ReadVariableOp_2:value:0&classifier/fc1/StopGradient_1:output:0*
T0*
_output_shapes

:@z
classifier/fc1/MatMulMatMulencoder_inputclassifier/fc1/add_3:z:0*
T0*'
_output_shapes
:���������X
classifier/fc1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :X
classifier/fc1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : ~
classifier/fc1/Pow_1Powclassifier/fc1/Pow_1/x:output:0classifier/fc1/Pow_1/y:output:0*
T0*
_output_shapes
: g
classifier/fc1/Cast_1Castclassifier/fc1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
classifier/fc1/ReadVariableOp_3ReadVariableOp(classifier_fc1_readvariableop_3_resource*
_output_shapes
:*
dtype0[
classifier/fc1/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
classifier/fc1/mul_4Mul'classifier/fc1/ReadVariableOp_3:value:0classifier/fc1/mul_4/y:output:0*
T0*
_output_shapes
:}
classifier/fc1/truediv_2RealDivclassifier/fc1/mul_4:z:0classifier/fc1/Cast_1:y:0*
T0*
_output_shapes
:^
classifier/fc1/Neg_2Negclassifier/fc1/truediv_2:z:0*
T0*
_output_shapes
:b
classifier/fc1/Round_1Roundclassifier/fc1/truediv_2:z:0*
T0*
_output_shapes
:x
classifier/fc1/add_4AddV2classifier/fc1/Neg_2:y:0classifier/fc1/Round_1:y:0*
T0*
_output_shapes
:l
classifier/fc1/StopGradient_2StopGradientclassifier/fc1/add_4:z:0*
T0*
_output_shapes
:�
classifier/fc1/add_5AddV2classifier/fc1/truediv_2:z:0&classifier/fc1/StopGradient_2:output:0*
T0*
_output_shapes
:m
(classifier/fc1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
&classifier/fc1/clip_by_value_1/MinimumMinimumclassifier/fc1/add_5:z:01classifier/fc1/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:e
 classifier/fc1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
classifier/fc1/clip_by_value_1Maximum*classifier/fc1/clip_by_value_1/Minimum:z:0)classifier/fc1/clip_by_value_1/y:output:0*
T0*
_output_shapes
:
classifier/fc1/mul_5Mulclassifier/fc1/Cast_1:y:0"classifier/fc1/clip_by_value_1:z:0*
T0*
_output_shapes
:_
classifier/fc1/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
classifier/fc1/truediv_3RealDivclassifier/fc1/mul_5:z:0#classifier/fc1/truediv_3/y:output:0*
T0*
_output_shapes
:[
classifier/fc1/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
classifier/fc1/mul_6Mulclassifier/fc1/mul_6/x:output:0classifier/fc1/truediv_3:z:0*
T0*
_output_shapes
:�
classifier/fc1/ReadVariableOp_4ReadVariableOp(classifier_fc1_readvariableop_3_resource*
_output_shapes
:*
dtype0i
classifier/fc1/Neg_3Neg'classifier/fc1/ReadVariableOp_4:value:0*
T0*
_output_shapes
:v
classifier/fc1/add_6AddV2classifier/fc1/Neg_3:y:0classifier/fc1/mul_6:z:0*
T0*
_output_shapes
:[
classifier/fc1/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{
classifier/fc1/mul_7Mulclassifier/fc1/mul_7/x:output:0classifier/fc1/add_6:z:0*
T0*
_output_shapes
:l
classifier/fc1/StopGradient_3StopGradientclassifier/fc1/mul_7:z:0*
T0*
_output_shapes
:�
classifier/fc1/ReadVariableOp_5ReadVariableOp(classifier_fc1_readvariableop_3_resource*
_output_shapes
:*
dtype0�
classifier/fc1/add_7AddV2'classifier/fc1/ReadVariableOp_5:value:0&classifier/fc1/StopGradient_3:output:0*
T0*
_output_shapes
:�
classifier/fc1/BiasAddBiasAddclassifier/fc1/MatMul:product:0classifier/fc1/add_7:z:0*
T0*'
_output_shapes
:���������X
classifier/relu1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :X
classifier/relu1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :~
classifier/relu1/PowPowclassifier/relu1/Pow/x:output:0classifier/relu1/Pow/y:output:0*
T0*
_output_shapes
: g
classifier/relu1/CastCastclassifier/relu1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
classifier/relu1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :Z
classifier/relu1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
classifier/relu1/Pow_1Pow!classifier/relu1/Pow_1/x:output:0!classifier/relu1/Pow_1/y:output:0*
T0*
_output_shapes
: k
classifier/relu1/Cast_1Castclassifier/relu1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: [
classifier/relu1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @[
classifier/relu1/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : s
classifier/relu1/Cast_2Cast"classifier/relu1/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: [
classifier/relu1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@z
classifier/relu1/subSubclassifier/relu1/Cast_2:y:0classifier/relu1/sub/y:output:0*
T0*
_output_shapes
: y
classifier/relu1/Pow_2Powclassifier/relu1/Const:output:0classifier/relu1/sub:z:0*
T0*
_output_shapes
: w
classifier/relu1/sub_1Subclassifier/relu1/Cast_1:y:0classifier/relu1/Pow_2:z:0*
T0*
_output_shapes
: �
classifier/relu1/LessEqual	LessEqualclassifier/fc1/BiasAdd:output:0classifier/relu1/sub_1:z:0*
T0*'
_output_shapes
:���������p
classifier/relu1/ReluReluclassifier/fc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������o
 classifier/relu1/ones_like/ShapeShapeclassifier/fc1/BiasAdd:output:0*
T0*
_output_shapes
:e
 classifier/relu1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu1/ones_likeFill)classifier/relu1/ones_like/Shape:output:0)classifier/relu1/ones_like/Const:output:0*
T0*'
_output_shapes
:���������w
classifier/relu1/sub_2Subclassifier/relu1/Cast_1:y:0classifier/relu1/Pow_2:z:0*
T0*
_output_shapes
: �
classifier/relu1/mulMul#classifier/relu1/ones_like:output:0classifier/relu1/sub_2:z:0*
T0*'
_output_shapes
:����������
classifier/relu1/SelectV2SelectV2classifier/relu1/LessEqual:z:0#classifier/relu1/Relu:activations:0classifier/relu1/mul:z:0*
T0*'
_output_shapes
:����������
classifier/relu1/mul_1Mulclassifier/fc1/BiasAdd:output:0classifier/relu1/Cast:y:0*
T0*'
_output_shapes
:����������
classifier/relu1/truedivRealDivclassifier/relu1/mul_1:z:0classifier/relu1/Cast_1:y:0*
T0*'
_output_shapes
:���������k
classifier/relu1/NegNegclassifier/relu1/truediv:z:0*
T0*'
_output_shapes
:���������o
classifier/relu1/RoundRoundclassifier/relu1/truediv:z:0*
T0*'
_output_shapes
:����������
classifier/relu1/addAddV2classifier/relu1/Neg:y:0classifier/relu1/Round:y:0*
T0*'
_output_shapes
:���������y
classifier/relu1/StopGradientStopGradientclassifier/relu1/add:z:0*
T0*'
_output_shapes
:����������
classifier/relu1/add_1AddV2classifier/relu1/truediv:z:0&classifier/relu1/StopGradient:output:0*
T0*'
_output_shapes
:����������
classifier/relu1/truediv_1RealDivclassifier/relu1/add_1:z:0classifier/relu1/Cast:y:0*
T0*'
_output_shapes
:���������a
classifier/relu1/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu1/truediv_2RealDiv%classifier/relu1/truediv_2/x:output:0classifier/relu1/Cast:y:0*
T0*
_output_shapes
: ]
classifier/relu1/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu1/sub_3Sub!classifier/relu1/sub_3/x:output:0classifier/relu1/truediv_2:z:0*
T0*
_output_shapes
: �
&classifier/relu1/clip_by_value/MinimumMinimumclassifier/relu1/truediv_1:z:0classifier/relu1/sub_3:z:0*
T0*'
_output_shapes
:���������e
 classifier/relu1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
classifier/relu1/clip_by_valueMaximum*classifier/relu1/clip_by_value/Minimum:z:0)classifier/relu1/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
classifier/relu1/mul_2Mulclassifier/relu1/Cast_1:y:0"classifier/relu1/clip_by_value:z:0*
T0*'
_output_shapes
:���������s
classifier/relu1/Neg_1Neg"classifier/relu1/SelectV2:output:0*
T0*'
_output_shapes
:����������
classifier/relu1/add_2AddV2classifier/relu1/Neg_1:y:0classifier/relu1/mul_2:z:0*
T0*'
_output_shapes
:���������]
classifier/relu1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu1/mul_3Mul!classifier/relu1/mul_3/x:output:0classifier/relu1/add_2:z:0*
T0*'
_output_shapes
:���������}
classifier/relu1/StopGradient_1StopGradientclassifier/relu1/mul_3:z:0*
T0*'
_output_shapes
:����������
classifier/relu1/add_3AddV2"classifier/relu1/SelectV2:output:0(classifier/relu1/StopGradient_1:output:0*
T0*'
_output_shapes
:���������O
1classifier/prune_low_magnitude_fc2_prun/no_updateNoOp*
_output_shapes
 Q
3classifier/prune_low_magnitude_fc2_prun/no_update_1NoOp*
_output_shapes
 �
:classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOpReadVariableOpCclassifier_prune_low_magnitude_fc2_prun_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
<classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1ReadVariableOpEclassifier_prune_low_magnitude_fc2_prun_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
+classifier/prune_low_magnitude_fc2_prun/MulMulBclassifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp:value:0Dclassifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
8classifier/prune_low_magnitude_fc2_prun/AssignVariableOpAssignVariableOpCclassifier_prune_low_magnitude_fc2_prun_mul_readvariableop_resource/classifier/prune_low_magnitude_fc2_prun/Mul:z:0;^classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
2classifier/prune_low_magnitude_fc2_prun/group_depsNoOp9^classifier/prune_low_magnitude_fc2_prun/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
4classifier/prune_low_magnitude_fc2_prun/group_deps_1NoOp3^classifier/prune_low_magnitude_fc2_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 o
-classifier/prune_low_magnitude_fc2_prun/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :o
-classifier/prune_low_magnitude_fc2_prun/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
+classifier/prune_low_magnitude_fc2_prun/PowPow6classifier/prune_low_magnitude_fc2_prun/Pow/x:output:06classifier/prune_low_magnitude_fc2_prun/Pow/y:output:0*
T0*
_output_shapes
: �
,classifier/prune_low_magnitude_fc2_prun/CastCast/classifier/prune_low_magnitude_fc2_prun/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
6classifier/prune_low_magnitude_fc2_prun/ReadVariableOpReadVariableOpCclassifier_prune_low_magnitude_fc2_prun_mul_readvariableop_resource9^classifier/prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes

:*
dtype0t
/classifier/prune_low_magnitude_fc2_prun/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
-classifier/prune_low_magnitude_fc2_prun/mul_1Mul>classifier/prune_low_magnitude_fc2_prun/ReadVariableOp:value:08classifier/prune_low_magnitude_fc2_prun/mul_1/y:output:0*
T0*
_output_shapes

:�
/classifier/prune_low_magnitude_fc2_prun/truedivRealDiv1classifier/prune_low_magnitude_fc2_prun/mul_1:z:00classifier/prune_low_magnitude_fc2_prun/Cast:y:0*
T0*
_output_shapes

:�
+classifier/prune_low_magnitude_fc2_prun/NegNeg3classifier/prune_low_magnitude_fc2_prun/truediv:z:0*
T0*
_output_shapes

:�
-classifier/prune_low_magnitude_fc2_prun/RoundRound3classifier/prune_low_magnitude_fc2_prun/truediv:z:0*
T0*
_output_shapes

:�
+classifier/prune_low_magnitude_fc2_prun/addAddV2/classifier/prune_low_magnitude_fc2_prun/Neg:y:01classifier/prune_low_magnitude_fc2_prun/Round:y:0*
T0*
_output_shapes

:�
4classifier/prune_low_magnitude_fc2_prun/StopGradientStopGradient/classifier/prune_low_magnitude_fc2_prun/add:z:0*
T0*
_output_shapes

:�
-classifier/prune_low_magnitude_fc2_prun/add_1AddV23classifier/prune_low_magnitude_fc2_prun/truediv:z:0=classifier/prune_low_magnitude_fc2_prun/StopGradient:output:0*
T0*
_output_shapes

:�
?classifier/prune_low_magnitude_fc2_prun/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
=classifier/prune_low_magnitude_fc2_prun/clip_by_value/MinimumMinimum1classifier/prune_low_magnitude_fc2_prun/add_1:z:0Hclassifier/prune_low_magnitude_fc2_prun/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:|
7classifier/prune_low_magnitude_fc2_prun/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
5classifier/prune_low_magnitude_fc2_prun/clip_by_valueMaximumAclassifier/prune_low_magnitude_fc2_prun/clip_by_value/Minimum:z:0@classifier/prune_low_magnitude_fc2_prun/clip_by_value/y:output:0*
T0*
_output_shapes

:�
-classifier/prune_low_magnitude_fc2_prun/mul_2Mul0classifier/prune_low_magnitude_fc2_prun/Cast:y:09classifier/prune_low_magnitude_fc2_prun/clip_by_value:z:0*
T0*
_output_shapes

:x
3classifier/prune_low_magnitude_fc2_prun/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
1classifier/prune_low_magnitude_fc2_prun/truediv_1RealDiv1classifier/prune_low_magnitude_fc2_prun/mul_2:z:0<classifier/prune_low_magnitude_fc2_prun/truediv_1/y:output:0*
T0*
_output_shapes

:t
/classifier/prune_low_magnitude_fc2_prun/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-classifier/prune_low_magnitude_fc2_prun/mul_3Mul8classifier/prune_low_magnitude_fc2_prun/mul_3/x:output:05classifier/prune_low_magnitude_fc2_prun/truediv_1:z:0*
T0*
_output_shapes

:�
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_1ReadVariableOpCclassifier_prune_low_magnitude_fc2_prun_mul_readvariableop_resource9^classifier/prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
-classifier/prune_low_magnitude_fc2_prun/Neg_1Neg@classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
-classifier/prune_low_magnitude_fc2_prun/add_2AddV21classifier/prune_low_magnitude_fc2_prun/Neg_1:y:01classifier/prune_low_magnitude_fc2_prun/mul_3:z:0*
T0*
_output_shapes

:t
/classifier/prune_low_magnitude_fc2_prun/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-classifier/prune_low_magnitude_fc2_prun/mul_4Mul8classifier/prune_low_magnitude_fc2_prun/mul_4/x:output:01classifier/prune_low_magnitude_fc2_prun/add_2:z:0*
T0*
_output_shapes

:�
6classifier/prune_low_magnitude_fc2_prun/StopGradient_1StopGradient1classifier/prune_low_magnitude_fc2_prun/mul_4:z:0*
T0*
_output_shapes

:�
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_2ReadVariableOpCclassifier_prune_low_magnitude_fc2_prun_mul_readvariableop_resource9^classifier/prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
-classifier/prune_low_magnitude_fc2_prun/add_3AddV2@classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_2:value:0?classifier/prune_low_magnitude_fc2_prun/StopGradient_1:output:0*
T0*
_output_shapes

:�
.classifier/prune_low_magnitude_fc2_prun/MatMulMatMulclassifier/relu1/add_3:z:01classifier/prune_low_magnitude_fc2_prun/add_3:z:0*
T0*'
_output_shapes
:���������q
/classifier/prune_low_magnitude_fc2_prun/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :q
/classifier/prune_low_magnitude_fc2_prun/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
-classifier/prune_low_magnitude_fc2_prun/Pow_1Pow8classifier/prune_low_magnitude_fc2_prun/Pow_1/x:output:08classifier/prune_low_magnitude_fc2_prun/Pow_1/y:output:0*
T0*
_output_shapes
: �
.classifier/prune_low_magnitude_fc2_prun/Cast_1Cast1classifier/prune_low_magnitude_fc2_prun/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_3ReadVariableOpAclassifier_prune_low_magnitude_fc2_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0t
/classifier/prune_low_magnitude_fc2_prun/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
-classifier/prune_low_magnitude_fc2_prun/mul_5Mul@classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_3:value:08classifier/prune_low_magnitude_fc2_prun/mul_5/y:output:0*
T0*
_output_shapes
:�
1classifier/prune_low_magnitude_fc2_prun/truediv_2RealDiv1classifier/prune_low_magnitude_fc2_prun/mul_5:z:02classifier/prune_low_magnitude_fc2_prun/Cast_1:y:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc2_prun/Neg_2Neg5classifier/prune_low_magnitude_fc2_prun/truediv_2:z:0*
T0*
_output_shapes
:�
/classifier/prune_low_magnitude_fc2_prun/Round_1Round5classifier/prune_low_magnitude_fc2_prun/truediv_2:z:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc2_prun/add_4AddV21classifier/prune_low_magnitude_fc2_prun/Neg_2:y:03classifier/prune_low_magnitude_fc2_prun/Round_1:y:0*
T0*
_output_shapes
:�
6classifier/prune_low_magnitude_fc2_prun/StopGradient_2StopGradient1classifier/prune_low_magnitude_fc2_prun/add_4:z:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc2_prun/add_5AddV25classifier/prune_low_magnitude_fc2_prun/truediv_2:z:0?classifier/prune_low_magnitude_fc2_prun/StopGradient_2:output:0*
T0*
_output_shapes
:�
Aclassifier/prune_low_magnitude_fc2_prun/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
?classifier/prune_low_magnitude_fc2_prun/clip_by_value_1/MinimumMinimum1classifier/prune_low_magnitude_fc2_prun/add_5:z:0Jclassifier/prune_low_magnitude_fc2_prun/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:~
9classifier/prune_low_magnitude_fc2_prun/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
7classifier/prune_low_magnitude_fc2_prun/clip_by_value_1MaximumCclassifier/prune_low_magnitude_fc2_prun/clip_by_value_1/Minimum:z:0Bclassifier/prune_low_magnitude_fc2_prun/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc2_prun/mul_6Mul2classifier/prune_low_magnitude_fc2_prun/Cast_1:y:0;classifier/prune_low_magnitude_fc2_prun/clip_by_value_1:z:0*
T0*
_output_shapes
:x
3classifier/prune_low_magnitude_fc2_prun/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
1classifier/prune_low_magnitude_fc2_prun/truediv_3RealDiv1classifier/prune_low_magnitude_fc2_prun/mul_6:z:0<classifier/prune_low_magnitude_fc2_prun/truediv_3/y:output:0*
T0*
_output_shapes
:t
/classifier/prune_low_magnitude_fc2_prun/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-classifier/prune_low_magnitude_fc2_prun/mul_7Mul8classifier/prune_low_magnitude_fc2_prun/mul_7/x:output:05classifier/prune_low_magnitude_fc2_prun/truediv_3:z:0*
T0*
_output_shapes
:�
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_4ReadVariableOpAclassifier_prune_low_magnitude_fc2_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0�
-classifier/prune_low_magnitude_fc2_prun/Neg_3Neg@classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_4:value:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc2_prun/add_6AddV21classifier/prune_low_magnitude_fc2_prun/Neg_3:y:01classifier/prune_low_magnitude_fc2_prun/mul_7:z:0*
T0*
_output_shapes
:t
/classifier/prune_low_magnitude_fc2_prun/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-classifier/prune_low_magnitude_fc2_prun/mul_8Mul8classifier/prune_low_magnitude_fc2_prun/mul_8/x:output:01classifier/prune_low_magnitude_fc2_prun/add_6:z:0*
T0*
_output_shapes
:�
6classifier/prune_low_magnitude_fc2_prun/StopGradient_3StopGradient1classifier/prune_low_magnitude_fc2_prun/mul_8:z:0*
T0*
_output_shapes
:�
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_5ReadVariableOpAclassifier_prune_low_magnitude_fc2_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0�
-classifier/prune_low_magnitude_fc2_prun/add_7AddV2@classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_5:value:0?classifier/prune_low_magnitude_fc2_prun/StopGradient_3:output:0*
T0*
_output_shapes
:�
/classifier/prune_low_magnitude_fc2_prun/BiasAddBiasAdd8classifier/prune_low_magnitude_fc2_prun/MatMul:product:01classifier/prune_low_magnitude_fc2_prun/add_7:z:0*
T0*'
_output_shapes
:���������X
classifier/relu2/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :X
classifier/relu2/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :~
classifier/relu2/PowPowclassifier/relu2/Pow/x:output:0classifier/relu2/Pow/y:output:0*
T0*
_output_shapes
: g
classifier/relu2/CastCastclassifier/relu2/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
classifier/relu2/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :Z
classifier/relu2/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
classifier/relu2/Pow_1Pow!classifier/relu2/Pow_1/x:output:0!classifier/relu2/Pow_1/y:output:0*
T0*
_output_shapes
: k
classifier/relu2/Cast_1Castclassifier/relu2/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: [
classifier/relu2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @[
classifier/relu2/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : s
classifier/relu2/Cast_2Cast"classifier/relu2/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: [
classifier/relu2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@z
classifier/relu2/subSubclassifier/relu2/Cast_2:y:0classifier/relu2/sub/y:output:0*
T0*
_output_shapes
: y
classifier/relu2/Pow_2Powclassifier/relu2/Const:output:0classifier/relu2/sub:z:0*
T0*
_output_shapes
: w
classifier/relu2/sub_1Subclassifier/relu2/Cast_1:y:0classifier/relu2/Pow_2:z:0*
T0*
_output_shapes
: �
classifier/relu2/LessEqual	LessEqual8classifier/prune_low_magnitude_fc2_prun/BiasAdd:output:0classifier/relu2/sub_1:z:0*
T0*'
_output_shapes
:����������
classifier/relu2/ReluRelu8classifier/prune_low_magnitude_fc2_prun/BiasAdd:output:0*
T0*'
_output_shapes
:����������
 classifier/relu2/ones_like/ShapeShape8classifier/prune_low_magnitude_fc2_prun/BiasAdd:output:0*
T0*
_output_shapes
:e
 classifier/relu2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu2/ones_likeFill)classifier/relu2/ones_like/Shape:output:0)classifier/relu2/ones_like/Const:output:0*
T0*'
_output_shapes
:���������w
classifier/relu2/sub_2Subclassifier/relu2/Cast_1:y:0classifier/relu2/Pow_2:z:0*
T0*
_output_shapes
: �
classifier/relu2/mulMul#classifier/relu2/ones_like:output:0classifier/relu2/sub_2:z:0*
T0*'
_output_shapes
:����������
classifier/relu2/SelectV2SelectV2classifier/relu2/LessEqual:z:0#classifier/relu2/Relu:activations:0classifier/relu2/mul:z:0*
T0*'
_output_shapes
:����������
classifier/relu2/mul_1Mul8classifier/prune_low_magnitude_fc2_prun/BiasAdd:output:0classifier/relu2/Cast:y:0*
T0*'
_output_shapes
:����������
classifier/relu2/truedivRealDivclassifier/relu2/mul_1:z:0classifier/relu2/Cast_1:y:0*
T0*'
_output_shapes
:���������k
classifier/relu2/NegNegclassifier/relu2/truediv:z:0*
T0*'
_output_shapes
:���������o
classifier/relu2/RoundRoundclassifier/relu2/truediv:z:0*
T0*'
_output_shapes
:����������
classifier/relu2/addAddV2classifier/relu2/Neg:y:0classifier/relu2/Round:y:0*
T0*'
_output_shapes
:���������y
classifier/relu2/StopGradientStopGradientclassifier/relu2/add:z:0*
T0*'
_output_shapes
:����������
classifier/relu2/add_1AddV2classifier/relu2/truediv:z:0&classifier/relu2/StopGradient:output:0*
T0*'
_output_shapes
:����������
classifier/relu2/truediv_1RealDivclassifier/relu2/add_1:z:0classifier/relu2/Cast:y:0*
T0*'
_output_shapes
:���������a
classifier/relu2/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu2/truediv_2RealDiv%classifier/relu2/truediv_2/x:output:0classifier/relu2/Cast:y:0*
T0*
_output_shapes
: ]
classifier/relu2/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu2/sub_3Sub!classifier/relu2/sub_3/x:output:0classifier/relu2/truediv_2:z:0*
T0*
_output_shapes
: �
&classifier/relu2/clip_by_value/MinimumMinimumclassifier/relu2/truediv_1:z:0classifier/relu2/sub_3:z:0*
T0*'
_output_shapes
:���������e
 classifier/relu2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
classifier/relu2/clip_by_valueMaximum*classifier/relu2/clip_by_value/Minimum:z:0)classifier/relu2/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
classifier/relu2/mul_2Mulclassifier/relu2/Cast_1:y:0"classifier/relu2/clip_by_value:z:0*
T0*'
_output_shapes
:���������s
classifier/relu2/Neg_1Neg"classifier/relu2/SelectV2:output:0*
T0*'
_output_shapes
:����������
classifier/relu2/add_2AddV2classifier/relu2/Neg_1:y:0classifier/relu2/mul_2:z:0*
T0*'
_output_shapes
:���������]
classifier/relu2/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu2/mul_3Mul!classifier/relu2/mul_3/x:output:0classifier/relu2/add_2:z:0*
T0*'
_output_shapes
:���������}
classifier/relu2/StopGradient_1StopGradientclassifier/relu2/mul_3:z:0*
T0*'
_output_shapes
:����������
classifier/relu2/add_3AddV2"classifier/relu2/SelectV2:output:0(classifier/relu2/StopGradient_1:output:0*
T0*'
_output_shapes
:���������O
1classifier/prune_low_magnitude_fc3_prun/no_updateNoOp*
_output_shapes
 Q
3classifier/prune_low_magnitude_fc3_prun/no_update_1NoOp*
_output_shapes
 �
:classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOpReadVariableOpCclassifier_prune_low_magnitude_fc3_prun_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
<classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1ReadVariableOpEclassifier_prune_low_magnitude_fc3_prun_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
+classifier/prune_low_magnitude_fc3_prun/MulMulBclassifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp:value:0Dclassifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
8classifier/prune_low_magnitude_fc3_prun/AssignVariableOpAssignVariableOpCclassifier_prune_low_magnitude_fc3_prun_mul_readvariableop_resource/classifier/prune_low_magnitude_fc3_prun/Mul:z:0;^classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
2classifier/prune_low_magnitude_fc3_prun/group_depsNoOp9^classifier/prune_low_magnitude_fc3_prun/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
4classifier/prune_low_magnitude_fc3_prun/group_deps_1NoOp3^classifier/prune_low_magnitude_fc3_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 o
-classifier/prune_low_magnitude_fc3_prun/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :o
-classifier/prune_low_magnitude_fc3_prun/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
+classifier/prune_low_magnitude_fc3_prun/PowPow6classifier/prune_low_magnitude_fc3_prun/Pow/x:output:06classifier/prune_low_magnitude_fc3_prun/Pow/y:output:0*
T0*
_output_shapes
: �
,classifier/prune_low_magnitude_fc3_prun/CastCast/classifier/prune_low_magnitude_fc3_prun/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
6classifier/prune_low_magnitude_fc3_prun/ReadVariableOpReadVariableOpCclassifier_prune_low_magnitude_fc3_prun_mul_readvariableop_resource9^classifier/prune_low_magnitude_fc3_prun/AssignVariableOp*
_output_shapes

:*
dtype0t
/classifier/prune_low_magnitude_fc3_prun/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
-classifier/prune_low_magnitude_fc3_prun/mul_1Mul>classifier/prune_low_magnitude_fc3_prun/ReadVariableOp:value:08classifier/prune_low_magnitude_fc3_prun/mul_1/y:output:0*
T0*
_output_shapes

:�
/classifier/prune_low_magnitude_fc3_prun/truedivRealDiv1classifier/prune_low_magnitude_fc3_prun/mul_1:z:00classifier/prune_low_magnitude_fc3_prun/Cast:y:0*
T0*
_output_shapes

:�
+classifier/prune_low_magnitude_fc3_prun/NegNeg3classifier/prune_low_magnitude_fc3_prun/truediv:z:0*
T0*
_output_shapes

:�
-classifier/prune_low_magnitude_fc3_prun/RoundRound3classifier/prune_low_magnitude_fc3_prun/truediv:z:0*
T0*
_output_shapes

:�
+classifier/prune_low_magnitude_fc3_prun/addAddV2/classifier/prune_low_magnitude_fc3_prun/Neg:y:01classifier/prune_low_magnitude_fc3_prun/Round:y:0*
T0*
_output_shapes

:�
4classifier/prune_low_magnitude_fc3_prun/StopGradientStopGradient/classifier/prune_low_magnitude_fc3_prun/add:z:0*
T0*
_output_shapes

:�
-classifier/prune_low_magnitude_fc3_prun/add_1AddV23classifier/prune_low_magnitude_fc3_prun/truediv:z:0=classifier/prune_low_magnitude_fc3_prun/StopGradient:output:0*
T0*
_output_shapes

:�
?classifier/prune_low_magnitude_fc3_prun/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
=classifier/prune_low_magnitude_fc3_prun/clip_by_value/MinimumMinimum1classifier/prune_low_magnitude_fc3_prun/add_1:z:0Hclassifier/prune_low_magnitude_fc3_prun/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:|
7classifier/prune_low_magnitude_fc3_prun/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
5classifier/prune_low_magnitude_fc3_prun/clip_by_valueMaximumAclassifier/prune_low_magnitude_fc3_prun/clip_by_value/Minimum:z:0@classifier/prune_low_magnitude_fc3_prun/clip_by_value/y:output:0*
T0*
_output_shapes

:�
-classifier/prune_low_magnitude_fc3_prun/mul_2Mul0classifier/prune_low_magnitude_fc3_prun/Cast:y:09classifier/prune_low_magnitude_fc3_prun/clip_by_value:z:0*
T0*
_output_shapes

:x
3classifier/prune_low_magnitude_fc3_prun/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
1classifier/prune_low_magnitude_fc3_prun/truediv_1RealDiv1classifier/prune_low_magnitude_fc3_prun/mul_2:z:0<classifier/prune_low_magnitude_fc3_prun/truediv_1/y:output:0*
T0*
_output_shapes

:t
/classifier/prune_low_magnitude_fc3_prun/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-classifier/prune_low_magnitude_fc3_prun/mul_3Mul8classifier/prune_low_magnitude_fc3_prun/mul_3/x:output:05classifier/prune_low_magnitude_fc3_prun/truediv_1:z:0*
T0*
_output_shapes

:�
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_1ReadVariableOpCclassifier_prune_low_magnitude_fc3_prun_mul_readvariableop_resource9^classifier/prune_low_magnitude_fc3_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
-classifier/prune_low_magnitude_fc3_prun/Neg_1Neg@classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
-classifier/prune_low_magnitude_fc3_prun/add_2AddV21classifier/prune_low_magnitude_fc3_prun/Neg_1:y:01classifier/prune_low_magnitude_fc3_prun/mul_3:z:0*
T0*
_output_shapes

:t
/classifier/prune_low_magnitude_fc3_prun/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-classifier/prune_low_magnitude_fc3_prun/mul_4Mul8classifier/prune_low_magnitude_fc3_prun/mul_4/x:output:01classifier/prune_low_magnitude_fc3_prun/add_2:z:0*
T0*
_output_shapes

:�
6classifier/prune_low_magnitude_fc3_prun/StopGradient_1StopGradient1classifier/prune_low_magnitude_fc3_prun/mul_4:z:0*
T0*
_output_shapes

:�
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_2ReadVariableOpCclassifier_prune_low_magnitude_fc3_prun_mul_readvariableop_resource9^classifier/prune_low_magnitude_fc3_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
-classifier/prune_low_magnitude_fc3_prun/add_3AddV2@classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_2:value:0?classifier/prune_low_magnitude_fc3_prun/StopGradient_1:output:0*
T0*
_output_shapes

:�
.classifier/prune_low_magnitude_fc3_prun/MatMulMatMulclassifier/relu2/add_3:z:01classifier/prune_low_magnitude_fc3_prun/add_3:z:0*
T0*'
_output_shapes
:���������q
/classifier/prune_low_magnitude_fc3_prun/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :q
/classifier/prune_low_magnitude_fc3_prun/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
-classifier/prune_low_magnitude_fc3_prun/Pow_1Pow8classifier/prune_low_magnitude_fc3_prun/Pow_1/x:output:08classifier/prune_low_magnitude_fc3_prun/Pow_1/y:output:0*
T0*
_output_shapes
: �
.classifier/prune_low_magnitude_fc3_prun/Cast_1Cast1classifier/prune_low_magnitude_fc3_prun/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_3ReadVariableOpAclassifier_prune_low_magnitude_fc3_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0t
/classifier/prune_low_magnitude_fc3_prun/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
-classifier/prune_low_magnitude_fc3_prun/mul_5Mul@classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_3:value:08classifier/prune_low_magnitude_fc3_prun/mul_5/y:output:0*
T0*
_output_shapes
:�
1classifier/prune_low_magnitude_fc3_prun/truediv_2RealDiv1classifier/prune_low_magnitude_fc3_prun/mul_5:z:02classifier/prune_low_magnitude_fc3_prun/Cast_1:y:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc3_prun/Neg_2Neg5classifier/prune_low_magnitude_fc3_prun/truediv_2:z:0*
T0*
_output_shapes
:�
/classifier/prune_low_magnitude_fc3_prun/Round_1Round5classifier/prune_low_magnitude_fc3_prun/truediv_2:z:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc3_prun/add_4AddV21classifier/prune_low_magnitude_fc3_prun/Neg_2:y:03classifier/prune_low_magnitude_fc3_prun/Round_1:y:0*
T0*
_output_shapes
:�
6classifier/prune_low_magnitude_fc3_prun/StopGradient_2StopGradient1classifier/prune_low_magnitude_fc3_prun/add_4:z:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc3_prun/add_5AddV25classifier/prune_low_magnitude_fc3_prun/truediv_2:z:0?classifier/prune_low_magnitude_fc3_prun/StopGradient_2:output:0*
T0*
_output_shapes
:�
Aclassifier/prune_low_magnitude_fc3_prun/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
?classifier/prune_low_magnitude_fc3_prun/clip_by_value_1/MinimumMinimum1classifier/prune_low_magnitude_fc3_prun/add_5:z:0Jclassifier/prune_low_magnitude_fc3_prun/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:~
9classifier/prune_low_magnitude_fc3_prun/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
7classifier/prune_low_magnitude_fc3_prun/clip_by_value_1MaximumCclassifier/prune_low_magnitude_fc3_prun/clip_by_value_1/Minimum:z:0Bclassifier/prune_low_magnitude_fc3_prun/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc3_prun/mul_6Mul2classifier/prune_low_magnitude_fc3_prun/Cast_1:y:0;classifier/prune_low_magnitude_fc3_prun/clip_by_value_1:z:0*
T0*
_output_shapes
:x
3classifier/prune_low_magnitude_fc3_prun/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
1classifier/prune_low_magnitude_fc3_prun/truediv_3RealDiv1classifier/prune_low_magnitude_fc3_prun/mul_6:z:0<classifier/prune_low_magnitude_fc3_prun/truediv_3/y:output:0*
T0*
_output_shapes
:t
/classifier/prune_low_magnitude_fc3_prun/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-classifier/prune_low_magnitude_fc3_prun/mul_7Mul8classifier/prune_low_magnitude_fc3_prun/mul_7/x:output:05classifier/prune_low_magnitude_fc3_prun/truediv_3:z:0*
T0*
_output_shapes
:�
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_4ReadVariableOpAclassifier_prune_low_magnitude_fc3_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0�
-classifier/prune_low_magnitude_fc3_prun/Neg_3Neg@classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_4:value:0*
T0*
_output_shapes
:�
-classifier/prune_low_magnitude_fc3_prun/add_6AddV21classifier/prune_low_magnitude_fc3_prun/Neg_3:y:01classifier/prune_low_magnitude_fc3_prun/mul_7:z:0*
T0*
_output_shapes
:t
/classifier/prune_low_magnitude_fc3_prun/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-classifier/prune_low_magnitude_fc3_prun/mul_8Mul8classifier/prune_low_magnitude_fc3_prun/mul_8/x:output:01classifier/prune_low_magnitude_fc3_prun/add_6:z:0*
T0*
_output_shapes
:�
6classifier/prune_low_magnitude_fc3_prun/StopGradient_3StopGradient1classifier/prune_low_magnitude_fc3_prun/mul_8:z:0*
T0*
_output_shapes
:�
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_5ReadVariableOpAclassifier_prune_low_magnitude_fc3_prun_readvariableop_3_resource*
_output_shapes
:*
dtype0�
-classifier/prune_low_magnitude_fc3_prun/add_7AddV2@classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_5:value:0?classifier/prune_low_magnitude_fc3_prun/StopGradient_3:output:0*
T0*
_output_shapes
:�
/classifier/prune_low_magnitude_fc3_prun/BiasAddBiasAdd8classifier/prune_low_magnitude_fc3_prun/MatMul:product:01classifier/prune_low_magnitude_fc3_prun/add_7:z:0*
T0*'
_output_shapes
:���������\
classifier/relu3_enc/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :\
classifier/relu3_enc/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
classifier/relu3_enc/PowPow#classifier/relu3_enc/Pow/x:output:0#classifier/relu3_enc/Pow/y:output:0*
T0*
_output_shapes
: o
classifier/relu3_enc/CastCastclassifier/relu3_enc/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: ^
classifier/relu3_enc/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :^
classifier/relu3_enc/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
classifier/relu3_enc/Pow_1Pow%classifier/relu3_enc/Pow_1/x:output:0%classifier/relu3_enc/Pow_1/y:output:0*
T0*
_output_shapes
: s
classifier/relu3_enc/Cast_1Castclassifier/relu3_enc/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: _
classifier/relu3_enc/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @_
classifier/relu3_enc/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : {
classifier/relu3_enc/Cast_2Cast&classifier/relu3_enc/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: _
classifier/relu3_enc/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
classifier/relu3_enc/subSubclassifier/relu3_enc/Cast_2:y:0#classifier/relu3_enc/sub/y:output:0*
T0*
_output_shapes
: �
classifier/relu3_enc/Pow_2Pow#classifier/relu3_enc/Const:output:0classifier/relu3_enc/sub:z:0*
T0*
_output_shapes
: �
classifier/relu3_enc/sub_1Subclassifier/relu3_enc/Cast_1:y:0classifier/relu3_enc/Pow_2:z:0*
T0*
_output_shapes
: �
classifier/relu3_enc/LessEqual	LessEqual8classifier/prune_low_magnitude_fc3_prun/BiasAdd:output:0classifier/relu3_enc/sub_1:z:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/ReluRelu8classifier/prune_low_magnitude_fc3_prun/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$classifier/relu3_enc/ones_like/ShapeShape8classifier/prune_low_magnitude_fc3_prun/BiasAdd:output:0*
T0*
_output_shapes
:i
$classifier/relu3_enc/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu3_enc/ones_likeFill-classifier/relu3_enc/ones_like/Shape:output:0-classifier/relu3_enc/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/sub_2Subclassifier/relu3_enc/Cast_1:y:0classifier/relu3_enc/Pow_2:z:0*
T0*
_output_shapes
: �
classifier/relu3_enc/mulMul'classifier/relu3_enc/ones_like:output:0classifier/relu3_enc/sub_2:z:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/SelectV2SelectV2"classifier/relu3_enc/LessEqual:z:0'classifier/relu3_enc/Relu:activations:0classifier/relu3_enc/mul:z:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/mul_1Mul8classifier/prune_low_magnitude_fc3_prun/BiasAdd:output:0classifier/relu3_enc/Cast:y:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/truedivRealDivclassifier/relu3_enc/mul_1:z:0classifier/relu3_enc/Cast_1:y:0*
T0*'
_output_shapes
:���������s
classifier/relu3_enc/NegNeg classifier/relu3_enc/truediv:z:0*
T0*'
_output_shapes
:���������w
classifier/relu3_enc/RoundRound classifier/relu3_enc/truediv:z:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/addAddV2classifier/relu3_enc/Neg:y:0classifier/relu3_enc/Round:y:0*
T0*'
_output_shapes
:����������
!classifier/relu3_enc/StopGradientStopGradientclassifier/relu3_enc/add:z:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/add_1AddV2 classifier/relu3_enc/truediv:z:0*classifier/relu3_enc/StopGradient:output:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/truediv_1RealDivclassifier/relu3_enc/add_1:z:0classifier/relu3_enc/Cast:y:0*
T0*'
_output_shapes
:���������e
 classifier/relu3_enc/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu3_enc/truediv_2RealDiv)classifier/relu3_enc/truediv_2/x:output:0classifier/relu3_enc/Cast:y:0*
T0*
_output_shapes
: a
classifier/relu3_enc/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu3_enc/sub_3Sub%classifier/relu3_enc/sub_3/x:output:0"classifier/relu3_enc/truediv_2:z:0*
T0*
_output_shapes
: �
*classifier/relu3_enc/clip_by_value/MinimumMinimum"classifier/relu3_enc/truediv_1:z:0classifier/relu3_enc/sub_3:z:0*
T0*'
_output_shapes
:���������i
$classifier/relu3_enc/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
"classifier/relu3_enc/clip_by_valueMaximum.classifier/relu3_enc/clip_by_value/Minimum:z:0-classifier/relu3_enc/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/mul_2Mulclassifier/relu3_enc/Cast_1:y:0&classifier/relu3_enc/clip_by_value:z:0*
T0*'
_output_shapes
:���������{
classifier/relu3_enc/Neg_1Neg&classifier/relu3_enc/SelectV2:output:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/add_2AddV2classifier/relu3_enc/Neg_1:y:0classifier/relu3_enc/mul_2:z:0*
T0*'
_output_shapes
:���������a
classifier/relu3_enc/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/relu3_enc/mul_3Mul%classifier/relu3_enc/mul_3/x:output:0classifier/relu3_enc/add_2:z:0*
T0*'
_output_shapes
:����������
#classifier/relu3_enc/StopGradient_1StopGradientclassifier/relu3_enc/mul_3:z:0*
T0*'
_output_shapes
:����������
classifier/relu3_enc/add_3AddV2&classifier/relu3_enc/SelectV2:output:0,classifier/relu3_enc/StopGradient_1:output:0*
T0*'
_output_shapes
:����������
/classifier/encoder_output/MatMul/ReadVariableOpReadVariableOp8classifier_encoder_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
 classifier/encoder_output/MatMulMatMulclassifier/relu3_enc/add_3:z:07classifier/encoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0classifier/encoder_output/BiasAdd/ReadVariableOpReadVariableOp9classifier_encoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!classifier/encoder_output/BiasAddBiasAdd*classifier/encoder_output/MatMul:product:08classifier/encoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
classifier/encoder_output/ReluRelu*classifier/encoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������V
8classifier/prune_low_magnitude_fc4_prunedclass/no_updateNoOp*
_output_shapes
 X
:classifier/prune_low_magnitude_fc4_prunedclass/no_update_1NoOp*
_output_shapes
 �
Aclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOpReadVariableOpJclassifier_prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
Cclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1ReadVariableOpLclassifier_prune_low_magnitude_fc4_prunedclass_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
2classifier/prune_low_magnitude_fc4_prunedclass/MulMulIclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp:value:0Kclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
?classifier/prune_low_magnitude_fc4_prunedclass/AssignVariableOpAssignVariableOpJclassifier_prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource6classifier/prune_low_magnitude_fc4_prunedclass/Mul:z:0B^classifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
9classifier/prune_low_magnitude_fc4_prunedclass/group_depsNoOp@^classifier/prune_low_magnitude_fc4_prunedclass/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
;classifier/prune_low_magnitude_fc4_prunedclass/group_deps_1NoOp:^classifier/prune_low_magnitude_fc4_prunedclass/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 v
4classifier/prune_low_magnitude_fc4_prunedclass/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :v
4classifier/prune_low_magnitude_fc4_prunedclass/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
2classifier/prune_low_magnitude_fc4_prunedclass/PowPow=classifier/prune_low_magnitude_fc4_prunedclass/Pow/x:output:0=classifier/prune_low_magnitude_fc4_prunedclass/Pow/y:output:0*
T0*
_output_shapes
: �
3classifier/prune_low_magnitude_fc4_prunedclass/CastCast6classifier/prune_low_magnitude_fc4_prunedclass/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
=classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOpReadVariableOpJclassifier_prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource@^classifier/prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
_output_shapes

:*
dtype0{
6classifier/prune_low_magnitude_fc4_prunedclass/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
4classifier/prune_low_magnitude_fc4_prunedclass/mul_1MulEclassifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp:value:0?classifier/prune_low_magnitude_fc4_prunedclass/mul_1/y:output:0*
T0*
_output_shapes

:�
6classifier/prune_low_magnitude_fc4_prunedclass/truedivRealDiv8classifier/prune_low_magnitude_fc4_prunedclass/mul_1:z:07classifier/prune_low_magnitude_fc4_prunedclass/Cast:y:0*
T0*
_output_shapes

:�
2classifier/prune_low_magnitude_fc4_prunedclass/NegNeg:classifier/prune_low_magnitude_fc4_prunedclass/truediv:z:0*
T0*
_output_shapes

:�
4classifier/prune_low_magnitude_fc4_prunedclass/RoundRound:classifier/prune_low_magnitude_fc4_prunedclass/truediv:z:0*
T0*
_output_shapes

:�
2classifier/prune_low_magnitude_fc4_prunedclass/addAddV26classifier/prune_low_magnitude_fc4_prunedclass/Neg:y:08classifier/prune_low_magnitude_fc4_prunedclass/Round:y:0*
T0*
_output_shapes

:�
;classifier/prune_low_magnitude_fc4_prunedclass/StopGradientStopGradient6classifier/prune_low_magnitude_fc4_prunedclass/add:z:0*
T0*
_output_shapes

:�
4classifier/prune_low_magnitude_fc4_prunedclass/add_1AddV2:classifier/prune_low_magnitude_fc4_prunedclass/truediv:z:0Dclassifier/prune_low_magnitude_fc4_prunedclass/StopGradient:output:0*
T0*
_output_shapes

:�
Fclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
Dclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value/MinimumMinimum8classifier/prune_low_magnitude_fc4_prunedclass/add_1:z:0Oclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:�
>classifier/prune_low_magnitude_fc4_prunedclass/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
<classifier/prune_low_magnitude_fc4_prunedclass/clip_by_valueMaximumHclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value/Minimum:z:0Gclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value/y:output:0*
T0*
_output_shapes

:�
4classifier/prune_low_magnitude_fc4_prunedclass/mul_2Mul7classifier/prune_low_magnitude_fc4_prunedclass/Cast:y:0@classifier/prune_low_magnitude_fc4_prunedclass/clip_by_value:z:0*
T0*
_output_shapes

:
:classifier/prune_low_magnitude_fc4_prunedclass/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
8classifier/prune_low_magnitude_fc4_prunedclass/truediv_1RealDiv8classifier/prune_low_magnitude_fc4_prunedclass/mul_2:z:0Cclassifier/prune_low_magnitude_fc4_prunedclass/truediv_1/y:output:0*
T0*
_output_shapes

:{
6classifier/prune_low_magnitude_fc4_prunedclass/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
4classifier/prune_low_magnitude_fc4_prunedclass/mul_3Mul?classifier/prune_low_magnitude_fc4_prunedclass/mul_3/x:output:0<classifier/prune_low_magnitude_fc4_prunedclass/truediv_1:z:0*
T0*
_output_shapes

:�
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1ReadVariableOpJclassifier_prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource@^classifier/prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
_output_shapes

:*
dtype0�
4classifier/prune_low_magnitude_fc4_prunedclass/Neg_1NegGclassifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
4classifier/prune_low_magnitude_fc4_prunedclass/add_2AddV28classifier/prune_low_magnitude_fc4_prunedclass/Neg_1:y:08classifier/prune_low_magnitude_fc4_prunedclass/mul_3:z:0*
T0*
_output_shapes

:{
6classifier/prune_low_magnitude_fc4_prunedclass/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
4classifier/prune_low_magnitude_fc4_prunedclass/mul_4Mul?classifier/prune_low_magnitude_fc4_prunedclass/mul_4/x:output:08classifier/prune_low_magnitude_fc4_prunedclass/add_2:z:0*
T0*
_output_shapes

:�
=classifier/prune_low_magnitude_fc4_prunedclass/StopGradient_1StopGradient8classifier/prune_low_magnitude_fc4_prunedclass/mul_4:z:0*
T0*
_output_shapes

:�
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2ReadVariableOpJclassifier_prune_low_magnitude_fc4_prunedclass_mul_readvariableop_resource@^classifier/prune_low_magnitude_fc4_prunedclass/AssignVariableOp*
_output_shapes

:*
dtype0�
4classifier/prune_low_magnitude_fc4_prunedclass/add_3AddV2Gclassifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2:value:0Fclassifier/prune_low_magnitude_fc4_prunedclass/StopGradient_1:output:0*
T0*
_output_shapes

:�
5classifier/prune_low_magnitude_fc4_prunedclass/MatMulMatMul,classifier/encoder_output/Relu:activations:08classifier/prune_low_magnitude_fc4_prunedclass/add_3:z:0*
T0*'
_output_shapes
:���������x
6classifier/prune_low_magnitude_fc4_prunedclass/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :x
6classifier/prune_low_magnitude_fc4_prunedclass/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
4classifier/prune_low_magnitude_fc4_prunedclass/Pow_1Pow?classifier/prune_low_magnitude_fc4_prunedclass/Pow_1/x:output:0?classifier/prune_low_magnitude_fc4_prunedclass/Pow_1/y:output:0*
T0*
_output_shapes
: �
5classifier/prune_low_magnitude_fc4_prunedclass/Cast_1Cast8classifier/prune_low_magnitude_fc4_prunedclass/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3ReadVariableOpHclassifier_prune_low_magnitude_fc4_prunedclass_readvariableop_3_resource*
_output_shapes
:*
dtype0{
6classifier/prune_low_magnitude_fc4_prunedclass/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
4classifier/prune_low_magnitude_fc4_prunedclass/mul_5MulGclassifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3:value:0?classifier/prune_low_magnitude_fc4_prunedclass/mul_5/y:output:0*
T0*
_output_shapes
:�
8classifier/prune_low_magnitude_fc4_prunedclass/truediv_2RealDiv8classifier/prune_low_magnitude_fc4_prunedclass/mul_5:z:09classifier/prune_low_magnitude_fc4_prunedclass/Cast_1:y:0*
T0*
_output_shapes
:�
4classifier/prune_low_magnitude_fc4_prunedclass/Neg_2Neg<classifier/prune_low_magnitude_fc4_prunedclass/truediv_2:z:0*
T0*
_output_shapes
:�
6classifier/prune_low_magnitude_fc4_prunedclass/Round_1Round<classifier/prune_low_magnitude_fc4_prunedclass/truediv_2:z:0*
T0*
_output_shapes
:�
4classifier/prune_low_magnitude_fc4_prunedclass/add_4AddV28classifier/prune_low_magnitude_fc4_prunedclass/Neg_2:y:0:classifier/prune_low_magnitude_fc4_prunedclass/Round_1:y:0*
T0*
_output_shapes
:�
=classifier/prune_low_magnitude_fc4_prunedclass/StopGradient_2StopGradient8classifier/prune_low_magnitude_fc4_prunedclass/add_4:z:0*
T0*
_output_shapes
:�
4classifier/prune_low_magnitude_fc4_prunedclass/add_5AddV2<classifier/prune_low_magnitude_fc4_prunedclass/truediv_2:z:0Fclassifier/prune_low_magnitude_fc4_prunedclass/StopGradient_2:output:0*
T0*
_output_shapes
:�
Hclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
Fclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value_1/MinimumMinimum8classifier/prune_low_magnitude_fc4_prunedclass/add_5:z:0Qclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:�
@classifier/prune_low_magnitude_fc4_prunedclass/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
>classifier/prune_low_magnitude_fc4_prunedclass/clip_by_value_1MaximumJclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value_1/Minimum:z:0Iclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
4classifier/prune_low_magnitude_fc4_prunedclass/mul_6Mul9classifier/prune_low_magnitude_fc4_prunedclass/Cast_1:y:0Bclassifier/prune_low_magnitude_fc4_prunedclass/clip_by_value_1:z:0*
T0*
_output_shapes
:
:classifier/prune_low_magnitude_fc4_prunedclass/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
8classifier/prune_low_magnitude_fc4_prunedclass/truediv_3RealDiv8classifier/prune_low_magnitude_fc4_prunedclass/mul_6:z:0Cclassifier/prune_low_magnitude_fc4_prunedclass/truediv_3/y:output:0*
T0*
_output_shapes
:{
6classifier/prune_low_magnitude_fc4_prunedclass/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
4classifier/prune_low_magnitude_fc4_prunedclass/mul_7Mul?classifier/prune_low_magnitude_fc4_prunedclass/mul_7/x:output:0<classifier/prune_low_magnitude_fc4_prunedclass/truediv_3:z:0*
T0*
_output_shapes
:�
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4ReadVariableOpHclassifier_prune_low_magnitude_fc4_prunedclass_readvariableop_3_resource*
_output_shapes
:*
dtype0�
4classifier/prune_low_magnitude_fc4_prunedclass/Neg_3NegGclassifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4:value:0*
T0*
_output_shapes
:�
4classifier/prune_low_magnitude_fc4_prunedclass/add_6AddV28classifier/prune_low_magnitude_fc4_prunedclass/Neg_3:y:08classifier/prune_low_magnitude_fc4_prunedclass/mul_7:z:0*
T0*
_output_shapes
:{
6classifier/prune_low_magnitude_fc4_prunedclass/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
4classifier/prune_low_magnitude_fc4_prunedclass/mul_8Mul?classifier/prune_low_magnitude_fc4_prunedclass/mul_8/x:output:08classifier/prune_low_magnitude_fc4_prunedclass/add_6:z:0*
T0*
_output_shapes
:�
=classifier/prune_low_magnitude_fc4_prunedclass/StopGradient_3StopGradient8classifier/prune_low_magnitude_fc4_prunedclass/mul_8:z:0*
T0*
_output_shapes
:�
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5ReadVariableOpHclassifier_prune_low_magnitude_fc4_prunedclass_readvariableop_3_resource*
_output_shapes
:*
dtype0�
4classifier/prune_low_magnitude_fc4_prunedclass/add_7AddV2Gclassifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5:value:0Fclassifier/prune_low_magnitude_fc4_prunedclass/StopGradient_3:output:0*
T0*
_output_shapes
:�
6classifier/prune_low_magnitude_fc4_prunedclass/BiasAddBiasAdd?classifier/prune_low_magnitude_fc4_prunedclass/MatMul:product:08classifier/prune_low_magnitude_fc4_prunedclass/add_7:z:0*
T0*'
_output_shapes
:���������b
 classifier/prunclass_relu4/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :b
 classifier/prunclass_relu4/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
classifier/prunclass_relu4/PowPow)classifier/prunclass_relu4/Pow/x:output:0)classifier/prunclass_relu4/Pow/y:output:0*
T0*
_output_shapes
: {
classifier/prunclass_relu4/CastCast"classifier/prunclass_relu4/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: d
"classifier/prunclass_relu4/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :d
"classifier/prunclass_relu4/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
 classifier/prunclass_relu4/Pow_1Pow+classifier/prunclass_relu4/Pow_1/x:output:0+classifier/prunclass_relu4/Pow_1/y:output:0*
T0*
_output_shapes
: 
!classifier/prunclass_relu4/Cast_1Cast$classifier/prunclass_relu4/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: e
 classifier/prunclass_relu4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
#classifier/prunclass_relu4/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : �
!classifier/prunclass_relu4/Cast_2Cast,classifier/prunclass_relu4/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: e
 classifier/prunclass_relu4/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
classifier/prunclass_relu4/subSub%classifier/prunclass_relu4/Cast_2:y:0)classifier/prunclass_relu4/sub/y:output:0*
T0*
_output_shapes
: �
 classifier/prunclass_relu4/Pow_2Pow)classifier/prunclass_relu4/Const:output:0"classifier/prunclass_relu4/sub:z:0*
T0*
_output_shapes
: �
 classifier/prunclass_relu4/sub_1Sub%classifier/prunclass_relu4/Cast_1:y:0$classifier/prunclass_relu4/Pow_2:z:0*
T0*
_output_shapes
: �
$classifier/prunclass_relu4/LessEqual	LessEqual?classifier/prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0$classifier/prunclass_relu4/sub_1:z:0*
T0*'
_output_shapes
:����������
classifier/prunclass_relu4/ReluRelu?classifier/prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0*
T0*'
_output_shapes
:����������
*classifier/prunclass_relu4/ones_like/ShapeShape?classifier/prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0*
T0*
_output_shapes
:o
*classifier/prunclass_relu4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$classifier/prunclass_relu4/ones_likeFill3classifier/prunclass_relu4/ones_like/Shape:output:03classifier/prunclass_relu4/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
 classifier/prunclass_relu4/sub_2Sub%classifier/prunclass_relu4/Cast_1:y:0$classifier/prunclass_relu4/Pow_2:z:0*
T0*
_output_shapes
: �
classifier/prunclass_relu4/mulMul-classifier/prunclass_relu4/ones_like:output:0$classifier/prunclass_relu4/sub_2:z:0*
T0*'
_output_shapes
:����������
#classifier/prunclass_relu4/SelectV2SelectV2(classifier/prunclass_relu4/LessEqual:z:0-classifier/prunclass_relu4/Relu:activations:0"classifier/prunclass_relu4/mul:z:0*
T0*'
_output_shapes
:����������
 classifier/prunclass_relu4/mul_1Mul?classifier/prune_low_magnitude_fc4_prunedclass/BiasAdd:output:0#classifier/prunclass_relu4/Cast:y:0*
T0*'
_output_shapes
:����������
"classifier/prunclass_relu4/truedivRealDiv$classifier/prunclass_relu4/mul_1:z:0%classifier/prunclass_relu4/Cast_1:y:0*
T0*'
_output_shapes
:���������
classifier/prunclass_relu4/NegNeg&classifier/prunclass_relu4/truediv:z:0*
T0*'
_output_shapes
:����������
 classifier/prunclass_relu4/RoundRound&classifier/prunclass_relu4/truediv:z:0*
T0*'
_output_shapes
:����������
classifier/prunclass_relu4/addAddV2"classifier/prunclass_relu4/Neg:y:0$classifier/prunclass_relu4/Round:y:0*
T0*'
_output_shapes
:����������
'classifier/prunclass_relu4/StopGradientStopGradient"classifier/prunclass_relu4/add:z:0*
T0*'
_output_shapes
:����������
 classifier/prunclass_relu4/add_1AddV2&classifier/prunclass_relu4/truediv:z:00classifier/prunclass_relu4/StopGradient:output:0*
T0*'
_output_shapes
:����������
$classifier/prunclass_relu4/truediv_1RealDiv$classifier/prunclass_relu4/add_1:z:0#classifier/prunclass_relu4/Cast:y:0*
T0*'
_output_shapes
:���������k
&classifier/prunclass_relu4/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$classifier/prunclass_relu4/truediv_2RealDiv/classifier/prunclass_relu4/truediv_2/x:output:0#classifier/prunclass_relu4/Cast:y:0*
T0*
_output_shapes
: g
"classifier/prunclass_relu4/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 classifier/prunclass_relu4/sub_3Sub+classifier/prunclass_relu4/sub_3/x:output:0(classifier/prunclass_relu4/truediv_2:z:0*
T0*
_output_shapes
: �
0classifier/prunclass_relu4/clip_by_value/MinimumMinimum(classifier/prunclass_relu4/truediv_1:z:0$classifier/prunclass_relu4/sub_3:z:0*
T0*'
_output_shapes
:���������o
*classifier/prunclass_relu4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(classifier/prunclass_relu4/clip_by_valueMaximum4classifier/prunclass_relu4/clip_by_value/Minimum:z:03classifier/prunclass_relu4/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
 classifier/prunclass_relu4/mul_2Mul%classifier/prunclass_relu4/Cast_1:y:0,classifier/prunclass_relu4/clip_by_value:z:0*
T0*'
_output_shapes
:����������
 classifier/prunclass_relu4/Neg_1Neg,classifier/prunclass_relu4/SelectV2:output:0*
T0*'
_output_shapes
:����������
 classifier/prunclass_relu4/add_2AddV2$classifier/prunclass_relu4/Neg_1:y:0$classifier/prunclass_relu4/mul_2:z:0*
T0*'
_output_shapes
:���������g
"classifier/prunclass_relu4/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 classifier/prunclass_relu4/mul_3Mul+classifier/prunclass_relu4/mul_3/x:output:0$classifier/prunclass_relu4/add_2:z:0*
T0*'
_output_shapes
:����������
)classifier/prunclass_relu4/StopGradient_1StopGradient$classifier/prunclass_relu4/mul_3:z:0*
T0*'
_output_shapes
:����������
 classifier/prunclass_relu4/add_3AddV2,classifier/prunclass_relu4/SelectV2:output:02classifier/prunclass_relu4/StopGradient_1:output:0*
T0*'
_output_shapes
:���������\
classifier/fc5_class/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :\
classifier/fc5_class/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : �
classifier/fc5_class/PowPow#classifier/fc5_class/Pow/x:output:0#classifier/fc5_class/Pow/y:output:0*
T0*
_output_shapes
: o
classifier/fc5_class/CastCastclassifier/fc5_class/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
#classifier/fc5_class/ReadVariableOpReadVariableOp,classifier_fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0_
classifier/fc5_class/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
classifier/fc5_class/mulMul+classifier/fc5_class/ReadVariableOp:value:0#classifier/fc5_class/mul/y:output:0*
T0*
_output_shapes

:(�
classifier/fc5_class/truedivRealDivclassifier/fc5_class/mul:z:0classifier/fc5_class/Cast:y:0*
T0*
_output_shapes

:(j
classifier/fc5_class/NegNeg classifier/fc5_class/truediv:z:0*
T0*
_output_shapes

:(n
classifier/fc5_class/RoundRound classifier/fc5_class/truediv:z:0*
T0*
_output_shapes

:(�
classifier/fc5_class/addAddV2classifier/fc5_class/Neg:y:0classifier/fc5_class/Round:y:0*
T0*
_output_shapes

:(x
!classifier/fc5_class/StopGradientStopGradientclassifier/fc5_class/add:z:0*
T0*
_output_shapes

:(�
classifier/fc5_class/add_1AddV2 classifier/fc5_class/truediv:z:0*classifier/fc5_class/StopGradient:output:0*
T0*
_output_shapes

:(q
,classifier/fc5_class/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
*classifier/fc5_class/clip_by_value/MinimumMinimumclassifier/fc5_class/add_1:z:05classifier/fc5_class/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(i
$classifier/fc5_class/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
"classifier/fc5_class/clip_by_valueMaximum.classifier/fc5_class/clip_by_value/Minimum:z:0-classifier/fc5_class/clip_by_value/y:output:0*
T0*
_output_shapes

:(�
classifier/fc5_class/mul_1Mulclassifier/fc5_class/Cast:y:0&classifier/fc5_class/clip_by_value:z:0*
T0*
_output_shapes

:(e
 classifier/fc5_class/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
classifier/fc5_class/truediv_1RealDivclassifier/fc5_class/mul_1:z:0)classifier/fc5_class/truediv_1/y:output:0*
T0*
_output_shapes

:(a
classifier/fc5_class/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/fc5_class/mul_2Mul%classifier/fc5_class/mul_2/x:output:0"classifier/fc5_class/truediv_1:z:0*
T0*
_output_shapes

:(�
%classifier/fc5_class/ReadVariableOp_1ReadVariableOp,classifier_fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0y
classifier/fc5_class/Neg_1Neg-classifier/fc5_class/ReadVariableOp_1:value:0*
T0*
_output_shapes

:(�
classifier/fc5_class/add_2AddV2classifier/fc5_class/Neg_1:y:0classifier/fc5_class/mul_2:z:0*
T0*
_output_shapes

:(a
classifier/fc5_class/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/fc5_class/mul_3Mul%classifier/fc5_class/mul_3/x:output:0classifier/fc5_class/add_2:z:0*
T0*
_output_shapes

:(|
#classifier/fc5_class/StopGradient_1StopGradientclassifier/fc5_class/mul_3:z:0*
T0*
_output_shapes

:(�
%classifier/fc5_class/ReadVariableOp_2ReadVariableOp,classifier_fc5_class_readvariableop_resource*
_output_shapes

:(*
dtype0�
classifier/fc5_class/add_3AddV2-classifier/fc5_class/ReadVariableOp_2:value:0,classifier/fc5_class/StopGradient_1:output:0*
T0*
_output_shapes

:(�
classifier/fc5_class/MatMulMatMul$classifier/prunclass_relu4/add_3:z:0classifier/fc5_class/add_3:z:0*
T0*'
_output_shapes
:���������(^
classifier/fc5_class/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :^
classifier/fc5_class/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
classifier/fc5_class/Pow_1Pow%classifier/fc5_class/Pow_1/x:output:0%classifier/fc5_class/Pow_1/y:output:0*
T0*
_output_shapes
: s
classifier/fc5_class/Cast_1Castclassifier/fc5_class/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
%classifier/fc5_class/ReadVariableOp_3ReadVariableOp.classifier_fc5_class_readvariableop_3_resource*
_output_shapes
:(*
dtype0a
classifier/fc5_class/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
classifier/fc5_class/mul_4Mul-classifier/fc5_class/ReadVariableOp_3:value:0%classifier/fc5_class/mul_4/y:output:0*
T0*
_output_shapes
:(�
classifier/fc5_class/truediv_2RealDivclassifier/fc5_class/mul_4:z:0classifier/fc5_class/Cast_1:y:0*
T0*
_output_shapes
:(j
classifier/fc5_class/Neg_2Neg"classifier/fc5_class/truediv_2:z:0*
T0*
_output_shapes
:(n
classifier/fc5_class/Round_1Round"classifier/fc5_class/truediv_2:z:0*
T0*
_output_shapes
:(�
classifier/fc5_class/add_4AddV2classifier/fc5_class/Neg_2:y:0 classifier/fc5_class/Round_1:y:0*
T0*
_output_shapes
:(x
#classifier/fc5_class/StopGradient_2StopGradientclassifier/fc5_class/add_4:z:0*
T0*
_output_shapes
:(�
classifier/fc5_class/add_5AddV2"classifier/fc5_class/truediv_2:z:0,classifier/fc5_class/StopGradient_2:output:0*
T0*
_output_shapes
:(s
.classifier/fc5_class/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
,classifier/fc5_class/clip_by_value_1/MinimumMinimumclassifier/fc5_class/add_5:z:07classifier/fc5_class/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(k
&classifier/fc5_class/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
$classifier/fc5_class/clip_by_value_1Maximum0classifier/fc5_class/clip_by_value_1/Minimum:z:0/classifier/fc5_class/clip_by_value_1/y:output:0*
T0*
_output_shapes
:(�
classifier/fc5_class/mul_5Mulclassifier/fc5_class/Cast_1:y:0(classifier/fc5_class/clip_by_value_1:z:0*
T0*
_output_shapes
:(e
 classifier/fc5_class/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B�
classifier/fc5_class/truediv_3RealDivclassifier/fc5_class/mul_5:z:0)classifier/fc5_class/truediv_3/y:output:0*
T0*
_output_shapes
:(a
classifier/fc5_class/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/fc5_class/mul_6Mul%classifier/fc5_class/mul_6/x:output:0"classifier/fc5_class/truediv_3:z:0*
T0*
_output_shapes
:(�
%classifier/fc5_class/ReadVariableOp_4ReadVariableOp.classifier_fc5_class_readvariableop_3_resource*
_output_shapes
:(*
dtype0u
classifier/fc5_class/Neg_3Neg-classifier/fc5_class/ReadVariableOp_4:value:0*
T0*
_output_shapes
:(�
classifier/fc5_class/add_6AddV2classifier/fc5_class/Neg_3:y:0classifier/fc5_class/mul_6:z:0*
T0*
_output_shapes
:(a
classifier/fc5_class/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/fc5_class/mul_7Mul%classifier/fc5_class/mul_7/x:output:0classifier/fc5_class/add_6:z:0*
T0*
_output_shapes
:(x
#classifier/fc5_class/StopGradient_3StopGradientclassifier/fc5_class/mul_7:z:0*
T0*
_output_shapes
:(�
%classifier/fc5_class/ReadVariableOp_5ReadVariableOp.classifier_fc5_class_readvariableop_3_resource*
_output_shapes
:(*
dtype0�
classifier/fc5_class/add_7AddV2-classifier/fc5_class/ReadVariableOp_5:value:0,classifier/fc5_class/StopGradient_3:output:0*
T0*
_output_shapes
:(�
classifier/fc5_class/BiasAddBiasAdd%classifier/fc5_class/MatMul:product:0classifier/fc5_class/add_7:z:0*
T0*'
_output_shapes
:���������(^
classifier/class_relu5/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :^
classifier/class_relu5/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
classifier/class_relu5/PowPow%classifier/class_relu5/Pow/x:output:0%classifier/class_relu5/Pow/y:output:0*
T0*
_output_shapes
: s
classifier/class_relu5/CastCastclassifier/class_relu5/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: `
classifier/class_relu5/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :`
classifier/class_relu5/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
classifier/class_relu5/Pow_1Pow'classifier/class_relu5/Pow_1/x:output:0'classifier/class_relu5/Pow_1/y:output:0*
T0*
_output_shapes
: w
classifier/class_relu5/Cast_1Cast classifier/class_relu5/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: a
classifier/class_relu5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @a
classifier/class_relu5/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : 
classifier/class_relu5/Cast_2Cast(classifier/class_relu5/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: a
classifier/class_relu5/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
classifier/class_relu5/subSub!classifier/class_relu5/Cast_2:y:0%classifier/class_relu5/sub/y:output:0*
T0*
_output_shapes
: �
classifier/class_relu5/Pow_2Pow%classifier/class_relu5/Const:output:0classifier/class_relu5/sub:z:0*
T0*
_output_shapes
: �
classifier/class_relu5/sub_1Sub!classifier/class_relu5/Cast_1:y:0 classifier/class_relu5/Pow_2:z:0*
T0*
_output_shapes
: �
 classifier/class_relu5/LessEqual	LessEqual%classifier/fc5_class/BiasAdd:output:0 classifier/class_relu5/sub_1:z:0*
T0*'
_output_shapes
:���������(|
classifier/class_relu5/ReluRelu%classifier/fc5_class/BiasAdd:output:0*
T0*'
_output_shapes
:���������({
&classifier/class_relu5/ones_like/ShapeShape%classifier/fc5_class/BiasAdd:output:0*
T0*
_output_shapes
:k
&classifier/class_relu5/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 classifier/class_relu5/ones_likeFill/classifier/class_relu5/ones_like/Shape:output:0/classifier/class_relu5/ones_like/Const:output:0*
T0*'
_output_shapes
:���������(�
classifier/class_relu5/sub_2Sub!classifier/class_relu5/Cast_1:y:0 classifier/class_relu5/Pow_2:z:0*
T0*
_output_shapes
: �
classifier/class_relu5/mulMul)classifier/class_relu5/ones_like:output:0 classifier/class_relu5/sub_2:z:0*
T0*'
_output_shapes
:���������(�
classifier/class_relu5/SelectV2SelectV2$classifier/class_relu5/LessEqual:z:0)classifier/class_relu5/Relu:activations:0classifier/class_relu5/mul:z:0*
T0*'
_output_shapes
:���������(�
classifier/class_relu5/mul_1Mul%classifier/fc5_class/BiasAdd:output:0classifier/class_relu5/Cast:y:0*
T0*'
_output_shapes
:���������(�
classifier/class_relu5/truedivRealDiv classifier/class_relu5/mul_1:z:0!classifier/class_relu5/Cast_1:y:0*
T0*'
_output_shapes
:���������(w
classifier/class_relu5/NegNeg"classifier/class_relu5/truediv:z:0*
T0*'
_output_shapes
:���������({
classifier/class_relu5/RoundRound"classifier/class_relu5/truediv:z:0*
T0*'
_output_shapes
:���������(�
classifier/class_relu5/addAddV2classifier/class_relu5/Neg:y:0 classifier/class_relu5/Round:y:0*
T0*'
_output_shapes
:���������(�
#classifier/class_relu5/StopGradientStopGradientclassifier/class_relu5/add:z:0*
T0*'
_output_shapes
:���������(�
classifier/class_relu5/add_1AddV2"classifier/class_relu5/truediv:z:0,classifier/class_relu5/StopGradient:output:0*
T0*'
_output_shapes
:���������(�
 classifier/class_relu5/truediv_1RealDiv classifier/class_relu5/add_1:z:0classifier/class_relu5/Cast:y:0*
T0*'
_output_shapes
:���������(g
"classifier/class_relu5/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 classifier/class_relu5/truediv_2RealDiv+classifier/class_relu5/truediv_2/x:output:0classifier/class_relu5/Cast:y:0*
T0*
_output_shapes
: c
classifier/class_relu5/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/class_relu5/sub_3Sub'classifier/class_relu5/sub_3/x:output:0$classifier/class_relu5/truediv_2:z:0*
T0*
_output_shapes
: �
,classifier/class_relu5/clip_by_value/MinimumMinimum$classifier/class_relu5/truediv_1:z:0 classifier/class_relu5/sub_3:z:0*
T0*'
_output_shapes
:���������(k
&classifier/class_relu5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
$classifier/class_relu5/clip_by_valueMaximum0classifier/class_relu5/clip_by_value/Minimum:z:0/classifier/class_relu5/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������(�
classifier/class_relu5/mul_2Mul!classifier/class_relu5/Cast_1:y:0(classifier/class_relu5/clip_by_value:z:0*
T0*'
_output_shapes
:���������(
classifier/class_relu5/Neg_1Neg(classifier/class_relu5/SelectV2:output:0*
T0*'
_output_shapes
:���������(�
classifier/class_relu5/add_2AddV2 classifier/class_relu5/Neg_1:y:0 classifier/class_relu5/mul_2:z:0*
T0*'
_output_shapes
:���������(c
classifier/class_relu5/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/class_relu5/mul_3Mul'classifier/class_relu5/mul_3/x:output:0 classifier/class_relu5/add_2:z:0*
T0*'
_output_shapes
:���������(�
%classifier/class_relu5/StopGradient_1StopGradient classifier/class_relu5/mul_3:z:0*
T0*'
_output_shapes
:���������(�
classifier/class_relu5/add_3AddV2(classifier/class_relu5/SelectV2:output:0.classifier/class_relu5/StopGradient_1:output:0*
T0*'
_output_shapes
:���������(a
classifier/classifier_out/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :a
classifier/classifier_out/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
classifier/classifier_out/PowPow(classifier/classifier_out/Pow/x:output:0(classifier/classifier_out/Pow/y:output:0*
T0*
_output_shapes
: y
classifier/classifier_out/CastCast!classifier/classifier_out/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
(classifier/classifier_out/ReadVariableOpReadVariableOp1classifier_classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0d
classifier/classifier_out/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier/classifier_out/mulMul0classifier/classifier_out/ReadVariableOp:value:0(classifier/classifier_out/mul/y:output:0*
T0*
_output_shapes

:(
�
!classifier/classifier_out/truedivRealDiv!classifier/classifier_out/mul:z:0"classifier/classifier_out/Cast:y:0*
T0*
_output_shapes

:(
t
classifier/classifier_out/NegNeg%classifier/classifier_out/truediv:z:0*
T0*
_output_shapes

:(
x
classifier/classifier_out/RoundRound%classifier/classifier_out/truediv:z:0*
T0*
_output_shapes

:(
�
classifier/classifier_out/addAddV2!classifier/classifier_out/Neg:y:0#classifier/classifier_out/Round:y:0*
T0*
_output_shapes

:(
�
&classifier/classifier_out/StopGradientStopGradient!classifier/classifier_out/add:z:0*
T0*
_output_shapes

:(
�
classifier/classifier_out/add_1AddV2%classifier/classifier_out/truediv:z:0/classifier/classifier_out/StopGradient:output:0*
T0*
_output_shapes

:(
v
1classifier/classifier_out/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��F�
/classifier/classifier_out/clip_by_value/MinimumMinimum#classifier/classifier_out/add_1:z:0:classifier/classifier_out/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(
n
)classifier/classifier_out/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
'classifier/classifier_out/clip_by_valueMaximum3classifier/classifier_out/clip_by_value/Minimum:z:02classifier/classifier_out/clip_by_value/y:output:0*
T0*
_output_shapes

:(
�
classifier/classifier_out/mul_1Mul"classifier/classifier_out/Cast:y:0+classifier/classifier_out/clip_by_value:z:0*
T0*
_output_shapes

:(
j
%classifier/classifier_out/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
#classifier/classifier_out/truediv_1RealDiv#classifier/classifier_out/mul_1:z:0.classifier/classifier_out/truediv_1/y:output:0*
T0*
_output_shapes

:(
f
!classifier/classifier_out/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/classifier_out/mul_2Mul*classifier/classifier_out/mul_2/x:output:0'classifier/classifier_out/truediv_1:z:0*
T0*
_output_shapes

:(
�
*classifier/classifier_out/ReadVariableOp_1ReadVariableOp1classifier_classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0�
classifier/classifier_out/Neg_1Neg2classifier/classifier_out/ReadVariableOp_1:value:0*
T0*
_output_shapes

:(
�
classifier/classifier_out/add_2AddV2#classifier/classifier_out/Neg_1:y:0#classifier/classifier_out/mul_2:z:0*
T0*
_output_shapes

:(
f
!classifier/classifier_out/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/classifier_out/mul_3Mul*classifier/classifier_out/mul_3/x:output:0#classifier/classifier_out/add_2:z:0*
T0*
_output_shapes

:(
�
(classifier/classifier_out/StopGradient_1StopGradient#classifier/classifier_out/mul_3:z:0*
T0*
_output_shapes

:(
�
*classifier/classifier_out/ReadVariableOp_2ReadVariableOp1classifier_classifier_out_readvariableop_resource*
_output_shapes

:(
*
dtype0�
classifier/classifier_out/add_3AddV22classifier/classifier_out/ReadVariableOp_2:value:01classifier/classifier_out/StopGradient_1:output:0*
T0*
_output_shapes

:(
�
 classifier/classifier_out/MatMulMatMul classifier/class_relu5/add_3:z:0#classifier/classifier_out/add_3:z:0*
T0*'
_output_shapes
:���������
c
!classifier/classifier_out/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :c
!classifier/classifier_out/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
classifier/classifier_out/Pow_1Pow*classifier/classifier_out/Pow_1/x:output:0*classifier/classifier_out/Pow_1/y:output:0*
T0*
_output_shapes
: }
 classifier/classifier_out/Cast_1Cast#classifier/classifier_out/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
*classifier/classifier_out/ReadVariableOp_3ReadVariableOp3classifier_classifier_out_readvariableop_3_resource*
_output_shapes
:
*
dtype0f
!classifier/classifier_out/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
classifier/classifier_out/mul_4Mul2classifier/classifier_out/ReadVariableOp_3:value:0*classifier/classifier_out/mul_4/y:output:0*
T0*
_output_shapes
:
�
#classifier/classifier_out/truediv_2RealDiv#classifier/classifier_out/mul_4:z:0$classifier/classifier_out/Cast_1:y:0*
T0*
_output_shapes
:
t
classifier/classifier_out/Neg_2Neg'classifier/classifier_out/truediv_2:z:0*
T0*
_output_shapes
:
x
!classifier/classifier_out/Round_1Round'classifier/classifier_out/truediv_2:z:0*
T0*
_output_shapes
:
�
classifier/classifier_out/add_4AddV2#classifier/classifier_out/Neg_2:y:0%classifier/classifier_out/Round_1:y:0*
T0*
_output_shapes
:
�
(classifier/classifier_out/StopGradient_2StopGradient#classifier/classifier_out/add_4:z:0*
T0*
_output_shapes
:
�
classifier/classifier_out/add_5AddV2'classifier/classifier_out/truediv_2:z:01classifier/classifier_out/StopGradient_2:output:0*
T0*
_output_shapes
:
x
3classifier/classifier_out/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��F�
1classifier/classifier_out/clip_by_value_1/MinimumMinimum#classifier/classifier_out/add_5:z:0<classifier/classifier_out/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:
p
+classifier/classifier_out/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
)classifier/classifier_out/clip_by_value_1Maximum5classifier/classifier_out/clip_by_value_1/Minimum:z:04classifier/classifier_out/clip_by_value_1/y:output:0*
T0*
_output_shapes
:
�
classifier/classifier_out/mul_5Mul$classifier/classifier_out/Cast_1:y:0-classifier/classifier_out/clip_by_value_1:z:0*
T0*
_output_shapes
:
j
%classifier/classifier_out/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G�
#classifier/classifier_out/truediv_3RealDiv#classifier/classifier_out/mul_5:z:0.classifier/classifier_out/truediv_3/y:output:0*
T0*
_output_shapes
:
f
!classifier/classifier_out/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/classifier_out/mul_6Mul*classifier/classifier_out/mul_6/x:output:0'classifier/classifier_out/truediv_3:z:0*
T0*
_output_shapes
:
�
*classifier/classifier_out/ReadVariableOp_4ReadVariableOp3classifier_classifier_out_readvariableop_3_resource*
_output_shapes
:
*
dtype0
classifier/classifier_out/Neg_3Neg2classifier/classifier_out/ReadVariableOp_4:value:0*
T0*
_output_shapes
:
�
classifier/classifier_out/add_6AddV2#classifier/classifier_out/Neg_3:y:0#classifier/classifier_out/mul_6:z:0*
T0*
_output_shapes
:
f
!classifier/classifier_out/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
classifier/classifier_out/mul_7Mul*classifier/classifier_out/mul_7/x:output:0#classifier/classifier_out/add_6:z:0*
T0*
_output_shapes
:
�
(classifier/classifier_out/StopGradient_3StopGradient#classifier/classifier_out/mul_7:z:0*
T0*
_output_shapes
:
�
*classifier/classifier_out/ReadVariableOp_5ReadVariableOp3classifier_classifier_out_readvariableop_3_resource*
_output_shapes
:
*
dtype0�
classifier/classifier_out/add_7AddV22classifier/classifier_out/ReadVariableOp_5:value:01classifier/classifier_out/StopGradient_3:output:0*
T0*
_output_shapes
:
�
!classifier/classifier_out/BiasAddBiasAdd*classifier/classifier_out/MatMul:product:0#classifier/classifier_out/add_7:z:0*
T0*'
_output_shapes
:���������
�
$classifier/classifier_output/SoftmaxSoftmax*classifier/classifier_out/BiasAdd:output:0*
T0*'
_output_shapes
:���������
}
IdentityIdentity.classifier/classifier_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp)^classifier/classifier_out/ReadVariableOp+^classifier/classifier_out/ReadVariableOp_1+^classifier/classifier_out/ReadVariableOp_2+^classifier/classifier_out/ReadVariableOp_3+^classifier/classifier_out/ReadVariableOp_4+^classifier/classifier_out/ReadVariableOp_51^classifier/encoder_output/BiasAdd/ReadVariableOp0^classifier/encoder_output/MatMul/ReadVariableOp^classifier/fc1/ReadVariableOp ^classifier/fc1/ReadVariableOp_1 ^classifier/fc1/ReadVariableOp_2 ^classifier/fc1/ReadVariableOp_3 ^classifier/fc1/ReadVariableOp_4 ^classifier/fc1/ReadVariableOp_5$^classifier/fc5_class/ReadVariableOp&^classifier/fc5_class/ReadVariableOp_1&^classifier/fc5_class/ReadVariableOp_2&^classifier/fc5_class/ReadVariableOp_3&^classifier/fc5_class/ReadVariableOp_4&^classifier/fc5_class/ReadVariableOp_59^classifier/prune_low_magnitude_fc2_prun/AssignVariableOp;^classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp=^classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_17^classifier/prune_low_magnitude_fc2_prun/ReadVariableOp9^classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_19^classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_29^classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_39^classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_49^classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_59^classifier/prune_low_magnitude_fc3_prun/AssignVariableOp;^classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp=^classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_17^classifier/prune_low_magnitude_fc3_prun/ReadVariableOp9^classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_19^classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_29^classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_39^classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_49^classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_5@^classifier/prune_low_magnitude_fc4_prunedclass/AssignVariableOpB^classifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOpD^classifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1>^classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp@^classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1@^classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2@^classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3@^classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4@^classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������@: : : : : : : : : : : : : : : : : 2T
(classifier/classifier_out/ReadVariableOp(classifier/classifier_out/ReadVariableOp2X
*classifier/classifier_out/ReadVariableOp_1*classifier/classifier_out/ReadVariableOp_12X
*classifier/classifier_out/ReadVariableOp_2*classifier/classifier_out/ReadVariableOp_22X
*classifier/classifier_out/ReadVariableOp_3*classifier/classifier_out/ReadVariableOp_32X
*classifier/classifier_out/ReadVariableOp_4*classifier/classifier_out/ReadVariableOp_42X
*classifier/classifier_out/ReadVariableOp_5*classifier/classifier_out/ReadVariableOp_52d
0classifier/encoder_output/BiasAdd/ReadVariableOp0classifier/encoder_output/BiasAdd/ReadVariableOp2b
/classifier/encoder_output/MatMul/ReadVariableOp/classifier/encoder_output/MatMul/ReadVariableOp2>
classifier/fc1/ReadVariableOpclassifier/fc1/ReadVariableOp2B
classifier/fc1/ReadVariableOp_1classifier/fc1/ReadVariableOp_12B
classifier/fc1/ReadVariableOp_2classifier/fc1/ReadVariableOp_22B
classifier/fc1/ReadVariableOp_3classifier/fc1/ReadVariableOp_32B
classifier/fc1/ReadVariableOp_4classifier/fc1/ReadVariableOp_42B
classifier/fc1/ReadVariableOp_5classifier/fc1/ReadVariableOp_52J
#classifier/fc5_class/ReadVariableOp#classifier/fc5_class/ReadVariableOp2N
%classifier/fc5_class/ReadVariableOp_1%classifier/fc5_class/ReadVariableOp_12N
%classifier/fc5_class/ReadVariableOp_2%classifier/fc5_class/ReadVariableOp_22N
%classifier/fc5_class/ReadVariableOp_3%classifier/fc5_class/ReadVariableOp_32N
%classifier/fc5_class/ReadVariableOp_4%classifier/fc5_class/ReadVariableOp_42N
%classifier/fc5_class/ReadVariableOp_5%classifier/fc5_class/ReadVariableOp_52t
8classifier/prune_low_magnitude_fc2_prun/AssignVariableOp8classifier/prune_low_magnitude_fc2_prun/AssignVariableOp2x
:classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp:classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2|
<classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1<classifier/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_12p
6classifier/prune_low_magnitude_fc2_prun/ReadVariableOp6classifier/prune_low_magnitude_fc2_prun/ReadVariableOp2t
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_18classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_12t
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_28classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_22t
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_38classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_32t
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_48classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_42t
8classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_58classifier/prune_low_magnitude_fc2_prun/ReadVariableOp_52t
8classifier/prune_low_magnitude_fc3_prun/AssignVariableOp8classifier/prune_low_magnitude_fc3_prun/AssignVariableOp2x
:classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp:classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp2|
<classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_1<classifier/prune_low_magnitude_fc3_prun/Mul/ReadVariableOp_12p
6classifier/prune_low_magnitude_fc3_prun/ReadVariableOp6classifier/prune_low_magnitude_fc3_prun/ReadVariableOp2t
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_18classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_12t
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_28classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_22t
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_38classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_32t
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_48classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_42t
8classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_58classifier/prune_low_magnitude_fc3_prun/ReadVariableOp_52�
?classifier/prune_low_magnitude_fc4_prunedclass/AssignVariableOp?classifier/prune_low_magnitude_fc4_prunedclass/AssignVariableOp2�
Aclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOpAclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp2�
Cclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_1Cclassifier/prune_low_magnitude_fc4_prunedclass/Mul/ReadVariableOp_12~
=classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp=classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp2�
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_1?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_12�
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_2?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_22�
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_3?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_32�
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_4?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_42�
?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5?classifier/prune_low_magnitude_fc4_prunedclass/ReadVariableOp_5:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�
G
+__inference_relu3_enc_layer_call_fn_1599439

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1595217`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
C
'__inference_relu2_layer_call_fn_1599085

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_1595086`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�>
�
cond_true_15962443
)cond_greaterequal_readvariableop_resource:	 >
,cond_pruning_ops_abs_readvariableop_resource:0
cond_assignvariableop_resource:*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
��cond/AssignVariableOp�cond/AssignVariableOp_1� cond/GreaterEqual/ReadVariableOp�cond/LessEqual/ReadVariableOp�cond/Sub/ReadVariableOp�#cond/pruning_ops/Abs/ReadVariableOp�
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	V
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	S
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N~
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: N
cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�NM
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : ^
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: V
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: `
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: y
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	M

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�f
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: Q
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rdb
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: N
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ^

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: \
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: O

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0q
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:X
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :�m
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: [
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?r
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: q
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: Z
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: _
cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
cond/pruning_ops/MaximumMaximumcond/pruning_ops/Round:y:0#cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: m
cond/pruning_ops/Cast_1Castcond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: q
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:�Z
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :��
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:�:�Z
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: `
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: Z
cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_2Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: b
 cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2_1GatherV2!cond/pruning_ops/TopKV2:indices:0cond/pruning_ops/sub_2:z:0)cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:Z
cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value
B :�`
cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zb
 cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
cond/pruning_ops/one_hotOneHot$cond/pruning_ops/GatherV2_1:output:0 cond/pruning_ops/Size_2:output:0'cond/pruning_ops/one_hot/Const:output:0)cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes	
:�q
 cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
cond/pruning_ops/Reshape_1Reshape!cond/pruning_ops/one_hot:output:0)cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
cond/pruning_ops/LogicalOr	LogicalOr!cond/pruning_ops/GreaterEqual:z:0#cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:i
	cond/CastCastcond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: `
cond/Identity_1Identitycond/Identity:output:0
^cond/NoOp*
T0
*
_output_shapes
: �
	cond/NoOpNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�
�
__inference_loss_fn_0_1600121J
8fc5_class_kernel_regularizer_abs_readvariableop_resource:(
identity��/fc5_class/kernel/Regularizer/Abs/ReadVariableOp�
/fc5_class/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8fc5_class_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:(*
dtype0�
 fc5_class/kernel/Regularizer/AbsAbs7fc5_class/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(s
"fc5_class/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 fc5_class/kernel/Regularizer/SumSum$fc5_class/kernel/Regularizer/Abs:y:0+fc5_class/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"fc5_class/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
 fc5_class/kernel/Regularizer/mulMul+fc5_class/kernel/Regularizer/mul/x:output:0)fc5_class/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$fc5_class/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^fc5_class/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/fc5_class/kernel/Regularizer/Abs/ReadVariableOp/fc5_class/kernel/Regularizer/Abs/ReadVariableOp
�
�
4assert_greater_equal_Assert_AssertGuard_true_1599631M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
r
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
I
-__inference_class_relu5_layer_call_fn_1599968

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_class_relu5_layer_call_and_return_conditional_losses_1595502`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������(:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
� 
h
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1599880

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
h
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1595371

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_1600132O
=classifier_out_kernel_regularizer_abs_readvariableop_resource:(

identity��4classifier_out/kernel/Regularizer/Abs/ReadVariableOp�
4classifier_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=classifier_out_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:(
*
dtype0�
%classifier_out/kernel/Regularizer/AbsAbs<classifier_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(
x
'classifier_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%classifier_out/kernel/Regularizer/SumSum)classifier_out/kernel/Regularizer/Abs:y:00classifier_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'classifier_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%classifier_out/kernel/Regularizer/mulMul0classifier_out/kernel/Regularizer/mul/x:output:0.classifier_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentity)classifier_out/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: }
NoOpNoOp5^classifier_out/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4classifier_out/kernel/Regularizer/Abs/ReadVariableOp4classifier_out/kernel/Regularizer/Abs/ReadVariableOp
�
j
N__inference_classifier_output_layer_call_and_return_conditional_losses_1595589

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�;
�
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1598878

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:'
readvariableop_3_resource:
identity��AssignVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5'
	no_updateNoOp*
_output_shapes
 )
no_update_1NoOp*
_output_shapes
 n
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype0r
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: }
ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A_
mul_1MulReadVariableOp:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
ReadVariableOp_1ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:
ReadVariableOp_2ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_5MulReadVariableOp_3:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
K__inference_encoder_output_layer_call_and_return_conditional_losses_1595230

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1595029

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:'
readvariableop_3_resource:
identity��AssignVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5'
	no_updateNoOp*
_output_shapes
 )
no_update_1NoOp*
_output_shapes
 n
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype0r
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: }
ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A_
mul_1MulReadVariableOp:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
ReadVariableOp_1ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:
ReadVariableOp_2ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_5MulReadVariableOp_3:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�G
�
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1599612

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:'
readvariableop_3_resource:
identity��AssignVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp'
	no_updateNoOp*
_output_shapes
 )
no_update_1NoOp*
_output_shapes
 n
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype0r
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: }
ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B_
mul_1MulReadVariableOp:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
ReadVariableOp_1ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:
ReadVariableOp_2ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_5MulReadVariableOp_3:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:����������
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5J^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�G
�
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1595314

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:'
readvariableop_3_resource:
identity��AssignVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp'
	no_updateNoOp*
_output_shapes
 )
no_update_1NoOp*
_output_shapes
 n
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype0r
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*
_output_shapes

:*
dtype0m
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 e
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: }
ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B_
mul_1MulReadVariableOp:value:0mul_1/y:output:0*
T0*
_output_shapes

:P
truedivRealDiv	mul_1:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_2MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B^
	truediv_1RealDiv	mul_2:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
ReadVariableOp_1ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:
ReadVariableOp_2ReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_5MulReadVariableOp_3:value:0mul_5/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_5:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_6Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_6:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_7Mulmul_7/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_7:z:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_8:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:����������
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5J^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Qprune_low_magnitude_fc3_prun_assert_greater_equal_Assert_AssertGuard_true_1597979�
�prune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc3_prun_assert_greater_equal_all
T
Pprune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_placeholder	V
Rprune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_placeholder_1	S
Oprune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_identity_1
�
Iprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Mprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc3_prun_assert_greater_equal_allJ^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Oprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityVprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "�
Oprune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_identity_1Xprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�k
�
5prune_low_magnitude_fc4_prunedclass_cond_true_1598269W
Mprune_low_magnitude_fc4_prunedclass_cond_greaterequal_readvariableop_resource:	 b
Pprune_low_magnitude_fc4_prunedclass_cond_pruning_ops_abs_readvariableop_resource:T
Bprune_low_magnitude_fc4_prunedclass_cond_assignvariableop_resource:N
Dprune_low_magnitude_fc4_prunedclass_cond_assignvariableop_1_resource: f
bprune_low_magnitude_fc4_prunedclass_cond_identity_prune_low_magnitude_fc4_prunedclass_logicaland_1
7
3prune_low_magnitude_fc4_prunedclass_cond_identity_1
��9prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp�;prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp_1�Dprune_low_magnitude_fc4_prunedclass/cond/GreaterEqual/ReadVariableOp�Aprune_low_magnitude_fc4_prunedclass/cond/LessEqual/ReadVariableOp�;prune_low_magnitude_fc4_prunedclass/cond/Sub/ReadVariableOp�Gprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Abs/ReadVariableOp�
Dprune_low_magnitude_fc4_prunedclass/cond/GreaterEqual/ReadVariableOpReadVariableOpMprune_low_magnitude_fc4_prunedclass_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	z
7prune_low_magnitude_fc4_prunedclass/cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
5prune_low_magnitude_fc4_prunedclass/cond/GreaterEqualGreaterEqualLprune_low_magnitude_fc4_prunedclass/cond/GreaterEqual/ReadVariableOp:value:0@prune_low_magnitude_fc4_prunedclass/cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
Aprune_low_magnitude_fc4_prunedclass/cond/LessEqual/ReadVariableOpReadVariableOpMprune_low_magnitude_fc4_prunedclass_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	w
4prune_low_magnitude_fc4_prunedclass/cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N�
2prune_low_magnitude_fc4_prunedclass/cond/LessEqual	LessEqualIprune_low_magnitude_fc4_prunedclass/cond/LessEqual/ReadVariableOp:value:0=prune_low_magnitude_fc4_prunedclass/cond/LessEqual/y:output:0*
T0	*
_output_shapes
: r
/prune_low_magnitude_fc4_prunedclass/cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�Nq
/prune_low_magnitude_fc4_prunedclass/cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : �
-prune_low_magnitude_fc4_prunedclass/cond/LessLess8prune_low_magnitude_fc4_prunedclass/cond/Less/x:output:08prune_low_magnitude_fc4_prunedclass/cond/Less/y:output:0*
T0*
_output_shapes
: �
2prune_low_magnitude_fc4_prunedclass/cond/LogicalOr	LogicalOr6prune_low_magnitude_fc4_prunedclass/cond/LessEqual:z:01prune_low_magnitude_fc4_prunedclass/cond/Less:z:0*
_output_shapes
: �
3prune_low_magnitude_fc4_prunedclass/cond/LogicalAnd
LogicalAnd9prune_low_magnitude_fc4_prunedclass/cond/GreaterEqual:z:06prune_low_magnitude_fc4_prunedclass/cond/LogicalOr:z:0*
_output_shapes
: �
;prune_low_magnitude_fc4_prunedclass/cond/Sub/ReadVariableOpReadVariableOpMprune_low_magnitude_fc4_prunedclass_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	q
.prune_low_magnitude_fc4_prunedclass/cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
,prune_low_magnitude_fc4_prunedclass/cond/SubSubCprune_low_magnitude_fc4_prunedclass/cond/Sub/ReadVariableOp:value:07prune_low_magnitude_fc4_prunedclass/cond/Sub/y:output:0*
T0	*
_output_shapes
: u
3prune_low_magnitude_fc4_prunedclass/cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd�
1prune_low_magnitude_fc4_prunedclass/cond/FloorModFloorMod0prune_low_magnitude_fc4_prunedclass/cond/Sub:z:0<prune_low_magnitude_fc4_prunedclass/cond/FloorMod/y:output:0*
T0	*
_output_shapes
: r
0prune_low_magnitude_fc4_prunedclass/cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
.prune_low_magnitude_fc4_prunedclass/cond/EqualEqual5prune_low_magnitude_fc4_prunedclass/cond/FloorMod:z:09prune_low_magnitude_fc4_prunedclass/cond/Equal/y:output:0*
T0	*
_output_shapes
: �
5prune_low_magnitude_fc4_prunedclass/cond/LogicalAnd_1
LogicalAnd7prune_low_magnitude_fc4_prunedclass/cond/LogicalAnd:z:02prune_low_magnitude_fc4_prunedclass/cond/Equal:z:0*
_output_shapes
: s
.prune_low_magnitude_fc4_prunedclass/cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
Gprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Abs/ReadVariableOpReadVariableOpPprune_low_magnitude_fc4_prunedclass_cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0�
8prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/AbsAbsOprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:{
9prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value	B : �
9prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/CastCastBprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 
:prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
8prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/subSubCprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub/x:output:07prune_low_magnitude_fc4_prunedclass/cond/Const:output:0*
T0*
_output_shapes
: �
8prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/mulMul=prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Cast:y:0<prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: �
:prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/RoundRound<prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/mul:z:0*
T0*
_output_shapes
: �
>prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
<prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/MaximumMaximum>prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Round:y:0Gprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: �
;prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Cast_1Cast@prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: �
Bprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
<prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/ReshapeReshape<prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Abs:y:0Kprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes
: }
;prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value	B : �
;prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/TopKV2TopKV2Eprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Reshape:output:0Dprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Size_1:output:0*
T0* 
_output_shapes
: : ~
<prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
:prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub_1Sub?prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Cast_1:y:0Eprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: �
Bprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GatherV2GatherV2Dprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/TopKV2:values:0>prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub_1:z:0Kprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: ~
<prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
:prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub_2Sub?prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Cast_1:y:0Eprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: �
Dprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GatherV2_1GatherV2Eprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/TopKV2:indices:0>prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/sub_2:z:0Mprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
Aprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GreaterEqualGreaterEqual<prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Abs:y:0Fprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:}
;prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value	B : �
Bprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z�
Dprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
<prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/one_hotOneHotHprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GatherV2_1:output:0Dprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Size_2:output:0Kprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/one_hot/Const:output:0Mprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes
: �
Dprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
>prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Reshape_1ReshapeEprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/one_hot:output:0Mprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
>prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/LogicalOr	LogicalOrEprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GreaterEqual:z:0Gprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:�
-prune_low_magnitude_fc4_prunedclass/cond/CastCastBprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
9prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOpAssignVariableOpBprune_low_magnitude_fc4_prunedclass_cond_assignvariableop_resource1prune_low_magnitude_fc4_prunedclass/cond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
;prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp_1AssignVariableOpDprune_low_magnitude_fc4_prunedclass_cond_assignvariableop_1_resourceFprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
3prune_low_magnitude_fc4_prunedclass/cond/group_depsNoOp:^prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp<^prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
1prune_low_magnitude_fc4_prunedclass/cond/IdentityIdentitybprune_low_magnitude_fc4_prunedclass_cond_identity_prune_low_magnitude_fc4_prunedclass_logicaland_14^prune_low_magnitude_fc4_prunedclass/cond/group_deps*
T0
*
_output_shapes
: �
3prune_low_magnitude_fc4_prunedclass/cond/Identity_1Identity:prune_low_magnitude_fc4_prunedclass/cond/Identity:output:0.^prune_low_magnitude_fc4_prunedclass/cond/NoOp*
T0
*
_output_shapes
: �
-prune_low_magnitude_fc4_prunedclass/cond/NoOpNoOp:^prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp<^prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp_1E^prune_low_magnitude_fc4_prunedclass/cond/GreaterEqual/ReadVariableOpB^prune_low_magnitude_fc4_prunedclass/cond/LessEqual/ReadVariableOp<^prune_low_magnitude_fc4_prunedclass/cond/Sub/ReadVariableOpH^prune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "s
3prune_low_magnitude_fc4_prunedclass_cond_identity_1<prune_low_magnitude_fc4_prunedclass/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2v
9prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp9prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp2z
;prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp_1;prune_low_magnitude_fc4_prunedclass/cond/AssignVariableOp_12�
Dprune_low_magnitude_fc4_prunedclass/cond/GreaterEqual/ReadVariableOpDprune_low_magnitude_fc4_prunedclass/cond/GreaterEqual/ReadVariableOp2�
Aprune_low_magnitude_fc4_prunedclass/cond/LessEqual/ReadVariableOpAprune_low_magnitude_fc4_prunedclass/cond/LessEqual/ReadVariableOp2z
;prune_low_magnitude_fc4_prunedclass/cond/Sub/ReadVariableOp;prune_low_magnitude_fc4_prunedclass/cond/Sub/ReadVariableOp2�
Gprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Abs/ReadVariableOpGprune_low_magnitude_fc4_prunedclass/cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�
�
,__inference_classifier_layer_call_fn_1596906

inputs
unknown:@
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:(

unknown_13:(

unknown_14:(


unknown_15:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_classifier_layer_call_and_return_conditional_losses_1595610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������@: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
� 
^
B__inference_relu2_layer_call_and_return_conditional_losses_1599134

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_1600143d
Rprune_low_magnitude_fc4_prunedclass_kernel_regularizer_abs_readvariableop_resource:
identity��Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpRprune_low_magnitude_fc4_prunedclass_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype0�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/AbsAbsQprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:�
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/SumSum>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs:y:0Eprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
:prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mulMulEprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul/x:output:0Cprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity>prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpJ^prune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Iprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOpIprune_low_magnitude_fc4_prunedclass/kernel/Regularizer/Abs/ReadVariableOp
�
�
E__inference_prune_low_magnitude_fc4_prunedclass_layer_call_fn_1599519

inputs
unknown:
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *i
fdRb
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1595314o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
5assert_greater_equal_Assert_AssertGuard_false_1595970K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
��.assert_greater_equal/Assert/AssertGuard/Assert�
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = �
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = �
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp/^assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_encoder_output_layer_call_fn_1599497

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_output_layer_call_and_return_conditional_losses_1595230o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
O
3__inference_classifier_output_layer_call_fn_1600105

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_classifier_output_layer_call_and_return_conditional_losses_1595589`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�>
�
cond_true_15996713
)cond_greaterequal_readvariableop_resource:	 >
,cond_pruning_ops_abs_readvariableop_resource:0
cond_assignvariableop_resource:*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
��cond/AssignVariableOp�cond/AssignVariableOp_1� cond/GreaterEqual/ReadVariableOp�cond/LessEqual/ReadVariableOp�cond/Sub/ReadVariableOp�#cond/pruning_ops/Abs/ReadVariableOp�
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	V
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	S
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N~
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: N
cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�NM
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : ^
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: V
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: `
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: y
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	M

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�f
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: Q
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rdb
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: N
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ^

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: \
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: O

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0q
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:W
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value	B : m
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: [
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?r
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: q
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: Z
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: _
cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
cond/pruning_ops/MaximumMaximumcond/pruning_ops/Round:y:0#cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: m
cond/pruning_ops/Cast_1Castcond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: q
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes
: Y
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0* 
_output_shapes
: : Z
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: `
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: Z
cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_2Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: b
 cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2_1GatherV2!cond/pruning_ops/TopKV2:indices:0cond/pruning_ops/sub_2:z:0)cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:Y
cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value	B : `
cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zb
 cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
cond/pruning_ops/one_hotOneHot$cond/pruning_ops/GatherV2_1:output:0 cond/pruning_ops/Size_2:output:0'cond/pruning_ops/one_hot/Const:output:0)cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes
: q
 cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
cond/pruning_ops/Reshape_1Reshape!cond/pruning_ops/one_hot:output:0)cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
cond/pruning_ops/LogicalOr	LogicalOr!cond/pruning_ops/GreaterEqual:z:0#cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:i
	cond/CastCastcond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: `
cond/Identity_1Identitycond/Identity:output:0
^cond/NoOp*
T0
*
_output_shapes
: �
	cond/NoOpNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�
�
E__inference_prune_low_magnitude_fc4_prunedclass_layer_call_fn_1599534

inputs
unknown:	 
	unknown_0:
	unknown_1:
	unknown_2: 
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *i
fdRb
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1595907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
>__inference_prune_low_magnitude_fc2_prun_layer_call_fn_1598791

inputs
unknown:
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *b
f]R[
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1595029o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_classifier_out_layer_call_fn_1600026

inputs
unknown:(

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_classifier_out_layer_call_and_return_conditional_losses_1595578o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�9
�
F__inference_fc5_class_layer_call_and_return_conditional_losses_1599963

inputs)
readvariableop_resource:('
readvariableop_3_resource:(
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�/fc5_class/kernel/Regularizer/Abs/ReadVariableOpG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:(N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:(@
NegNegtruediv:z:0*
T0*
_output_shapes

:(D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:(I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:(N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:([
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:(\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:(R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:(P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:(L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:(h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:(M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:(L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:(R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:(h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:(U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������(I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:(*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:(P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:(@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:(D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:(K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:(N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:([
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:(^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:(R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:(P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:(L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:(f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:(*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:(I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:(L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:(N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:(f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:(*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:(a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������(�
/fc5_class/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0�
 fc5_class/kernel/Regularizer/AbsAbs7fc5_class/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(s
"fc5_class/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 fc5_class/kernel/Regularizer/SumSum$fc5_class/kernel/Regularizer/Abs:y:0+fc5_class/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"fc5_class/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
 fc5_class/kernel/Regularizer/mulMul+fc5_class/kernel/Regularizer/mul/x:output:0)fc5_class/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_50^fc5_class/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52b
/fc5_class/kernel/Regularizer/Abs/ReadVariableOp/fc5_class/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�'
#__inference__traced_restore_1600518
file_prefix-
assignvariableop_fc1_kernel:@)
assignvariableop_1_fc1_bias:F
4assignvariableop_2_prune_low_magnitude_fc2_prun_mask:C
9assignvariableop_3_prune_low_magnitude_fc2_prun_threshold: F
<assignvariableop_4_prune_low_magnitude_fc2_prun_pruning_step:	 F
4assignvariableop_5_prune_low_magnitude_fc3_prun_mask:C
9assignvariableop_6_prune_low_magnitude_fc3_prun_threshold: F
<assignvariableop_7_prune_low_magnitude_fc3_prun_pruning_step:	 :
(assignvariableop_8_encoder_output_kernel:4
&assignvariableop_9_encoder_output_bias:N
<assignvariableop_10_prune_low_magnitude_fc4_prunedclass_mask:K
Aassignvariableop_11_prune_low_magnitude_fc4_prunedclass_threshold: N
Dassignvariableop_12_prune_low_magnitude_fc4_prunedclass_pruning_step:	 6
$assignvariableop_13_fc5_class_kernel:(0
"assignvariableop_14_fc5_class_bias:(;
)assignvariableop_15_classifier_out_kernel:(
5
'assignvariableop_16_classifier_out_bias:
I
7assignvariableop_17_prune_low_magnitude_fc2_prun_kernel:C
5assignvariableop_18_prune_low_magnitude_fc2_prun_bias:I
7assignvariableop_19_prune_low_magnitude_fc3_prun_kernel:C
5assignvariableop_20_prune_low_magnitude_fc3_prun_bias:P
>assignvariableop_21_prune_low_magnitude_fc4_prunedclass_kernel:J
<assignvariableop_22_prune_low_magnitude_fc4_prunedclass_bias:'
assignvariableop_23_iteration:	 +
!assignvariableop_24_learning_rate: 7
%assignvariableop_25_adam_m_fc1_kernel:@7
%assignvariableop_26_adam_v_fc1_kernel:@1
#assignvariableop_27_adam_m_fc1_bias:1
#assignvariableop_28_adam_v_fc1_bias:P
>assignvariableop_29_adam_m_prune_low_magnitude_fc2_prun_kernel:P
>assignvariableop_30_adam_v_prune_low_magnitude_fc2_prun_kernel:J
<assignvariableop_31_adam_m_prune_low_magnitude_fc2_prun_bias:J
<assignvariableop_32_adam_v_prune_low_magnitude_fc2_prun_bias:P
>assignvariableop_33_adam_m_prune_low_magnitude_fc3_prun_kernel:P
>assignvariableop_34_adam_v_prune_low_magnitude_fc3_prun_kernel:J
<assignvariableop_35_adam_m_prune_low_magnitude_fc3_prun_bias:J
<assignvariableop_36_adam_v_prune_low_magnitude_fc3_prun_bias:B
0assignvariableop_37_adam_m_encoder_output_kernel:B
0assignvariableop_38_adam_v_encoder_output_kernel:<
.assignvariableop_39_adam_m_encoder_output_bias:<
.assignvariableop_40_adam_v_encoder_output_bias:W
Eassignvariableop_41_adam_m_prune_low_magnitude_fc4_prunedclass_kernel:W
Eassignvariableop_42_adam_v_prune_low_magnitude_fc4_prunedclass_kernel:Q
Cassignvariableop_43_adam_m_prune_low_magnitude_fc4_prunedclass_bias:Q
Cassignvariableop_44_adam_v_prune_low_magnitude_fc4_prunedclass_bias:=
+assignvariableop_45_adam_m_fc5_class_kernel:(=
+assignvariableop_46_adam_v_fc5_class_kernel:(7
)assignvariableop_47_adam_m_fc5_class_bias:(7
)assignvariableop_48_adam_v_fc5_class_bias:(B
0assignvariableop_49_adam_m_classifier_out_kernel:(
B
0assignvariableop_50_adam_v_classifier_out_kernel:(
<
.assignvariableop_51_adam_m_classifier_out_bias:
<
.assignvariableop_52_adam_v_classifier_out_bias:
%
assignvariableop_53_total_1: %
assignvariableop_54_count_1: #
assignvariableop_55_total: #
assignvariableop_56_count: 
identity_58��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-2/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-4/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_fc1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_fc1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp4assignvariableop_2_prune_low_magnitude_fc2_prun_maskIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp9assignvariableop_3_prune_low_magnitude_fc2_prun_thresholdIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp<assignvariableop_4_prune_low_magnitude_fc2_prun_pruning_stepIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp4assignvariableop_5_prune_low_magnitude_fc3_prun_maskIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp9assignvariableop_6_prune_low_magnitude_fc3_prun_thresholdIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp<assignvariableop_7_prune_low_magnitude_fc3_prun_pruning_stepIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp(assignvariableop_8_encoder_output_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp&assignvariableop_9_encoder_output_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp<assignvariableop_10_prune_low_magnitude_fc4_prunedclass_maskIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpAassignvariableop_11_prune_low_magnitude_fc4_prunedclass_thresholdIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpDassignvariableop_12_prune_low_magnitude_fc4_prunedclass_pruning_stepIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_fc5_class_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_fc5_class_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_classifier_out_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_classifier_out_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp7assignvariableop_17_prune_low_magnitude_fc2_prun_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_prune_low_magnitude_fc2_prun_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp7assignvariableop_19_prune_low_magnitude_fc3_prun_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp5assignvariableop_20_prune_low_magnitude_fc3_prun_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp>assignvariableop_21_prune_low_magnitude_fc4_prunedclass_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp<assignvariableop_22_prune_low_magnitude_fc4_prunedclass_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_iterationIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_learning_rateIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_m_fc1_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_v_fc1_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp#assignvariableop_27_adam_m_fc1_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_adam_v_fc1_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_m_prune_low_magnitude_fc2_prun_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_v_prune_low_magnitude_fc2_prun_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp<assignvariableop_31_adam_m_prune_low_magnitude_fc2_prun_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp<assignvariableop_32_adam_v_prune_low_magnitude_fc2_prun_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp>assignvariableop_33_adam_m_prune_low_magnitude_fc3_prun_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adam_v_prune_low_magnitude_fc3_prun_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp<assignvariableop_35_adam_m_prune_low_magnitude_fc3_prun_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp<assignvariableop_36_adam_v_prune_low_magnitude_fc3_prun_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp0assignvariableop_37_adam_m_encoder_output_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp0assignvariableop_38_adam_v_encoder_output_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp.assignvariableop_39_adam_m_encoder_output_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_v_encoder_output_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpEassignvariableop_41_adam_m_prune_low_magnitude_fc4_prunedclass_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpEassignvariableop_42_adam_v_prune_low_magnitude_fc4_prunedclass_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpCassignvariableop_43_adam_m_prune_low_magnitude_fc4_prunedclass_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpCassignvariableop_44_adam_v_prune_low_magnitude_fc4_prunedclass_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_m_fc5_class_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_v_fc5_class_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_m_fc5_class_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_v_fc5_class_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp0assignvariableop_49_adam_m_classifier_out_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp0assignvariableop_50_adam_v_classifier_out_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp.assignvariableop_51_adam_m_classifier_out_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp.assignvariableop_52_adam_v_classifier_out_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_1Identity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_1Identity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpassignvariableop_55_totalIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_countIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*�
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
M
1__inference_prunclass_relu4_layer_call_fn_1599831

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1595371`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
cond_false_1596245
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
O
	cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 b
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: T
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
�
�
Rprune_low_magnitude_fc3_prun_assert_greater_equal_Assert_AssertGuard_false_1597980�
�prune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prun_assert_greater_equal_all
�
�prune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prun_assert_greater_equal_readvariableop	�
prune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prun_assert_greater_equal_y	S
Oprune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_identity_1
��Kprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert�
Rprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Rprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Rprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*Z
valueQBO BIx (prune_low_magnitude_fc3_prun/assert_greater_equal/ReadVariableOp:0) = �
Rprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (prune_low_magnitude_fc3_prun/assert_greater_equal/y:0) = �
Kprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/AssertAssert�prune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prun_assert_greater_equal_all[prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0[prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0[prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0�prune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prun_assert_greater_equal_readvariableop[prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0prune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prun_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Mprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prun_assert_greater_equal_allL^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
Oprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityVprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity:output:0J^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Iprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/NoOpNoOpL^prune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "�
Oprune_low_magnitude_fc3_prun_assert_greater_equal_assert_assertguard_identity_1Xprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2�
Kprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/AssertKprune_low_magnitude_fc3_prun/assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
� 
b
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1595217

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@G
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�1
�
@__inference_fc1_layer_call_and_return_conditional_losses_1598726

inputs)
readvariableop_resource:@'
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:@N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:@@
NegNegtruediv:z:0*
T0*
_output_shapes

:@D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:@I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:@N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:@[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:@R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:@P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:@L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:@h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:@M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:@L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:@R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:@h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:@U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  pAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�:
�
K__inference_classifier_out_layer_call_and_return_conditional_losses_1595578

inputs)
readvariableop_resource:(
'
readvariableop_3_resource:

identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�4classifier_out/kernel/Regularizer/Abs/ReadVariableOpG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(
*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:(
N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:(
@
NegNegtruediv:z:0*
T0*
_output_shapes

:(
D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:(
I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:(
N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:(
[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:(
\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Fv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:(
T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:(
R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:(
P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:(
L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:(
h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:(
*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:(
M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:(
L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:(
R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:(
h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:(
*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:(
U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������
I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:
*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   G]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:
P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:
@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:
D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:
K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:
N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:
[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:
^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Fv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:
V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:
R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:
P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   GZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:
L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:
f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:
*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:
I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:
L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:
N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:
f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:
*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:
a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������
�
4classifier_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(
*
dtype0�
%classifier_out/kernel/Regularizer/AbsAbs<classifier_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:(
x
'classifier_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%classifier_out/kernel/Regularizer/SumSum)classifier_out/kernel/Regularizer/Abs:y:00classifier_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: l
'classifier_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��8�
%classifier_out/kernel/Regularizer/mulMul0classifier_out/kernel/Regularizer/mul/x:output:0.classifier_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_55^classifier_out/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52l
4classifier_out/kernel/Regularizer/Abs/ReadVariableOp4classifier_out/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
4assert_greater_equal_Assert_AssertGuard_true_1598897M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
r
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
4assert_greater_equal_Assert_AssertGuard_true_1599251M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
r
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
cond_false_1596010
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
O
	cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 b
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: T
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
�>
�
cond_true_15992913
)cond_greaterequal_readvariableop_resource:	 >
,cond_pruning_ops_abs_readvariableop_resource:0
cond_assignvariableop_resource:*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
��cond/AssignVariableOp�cond/AssignVariableOp_1� cond/GreaterEqual/ReadVariableOp�cond/LessEqual/ReadVariableOp�cond/Sub/ReadVariableOp�#cond/pruning_ops/Abs/ReadVariableOp�
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	V
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	S
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N~
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: N
cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�NM
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : ^
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: V
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: `
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: y
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	M

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�f
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: Q
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rdb
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: N
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ^

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: \
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: O

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L?�
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0q
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:X
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :�m
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: [
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?r
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: q
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: Z
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: _
cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
cond/pruning_ops/MaximumMaximumcond/pruning_ops/Round:y:0#cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: m
cond/pruning_ops/Cast_1Castcond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: q
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:�Z
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :��
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:�:�Z
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: `
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: Z
cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :~
cond/pruning_ops/sub_2Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: b
 cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
cond/pruning_ops/GatherV2_1GatherV2!cond/pruning_ops/TopKV2:indices:0cond/pruning_ops/sub_2:z:0)cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:Z
cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value
B :�`
cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Zb
 cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
cond/pruning_ops/one_hotOneHot$cond/pruning_ops/GatherV2_1:output:0 cond/pruning_ops/Size_2:output:0'cond/pruning_ops/one_hot/Const:output:0)cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes	
:�q
 cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
cond/pruning_ops/Reshape_1Reshape!cond/pruning_ops/one_hot:output:0)cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
cond/pruning_ops/LogicalOr	LogicalOr!cond/pruning_ops/GreaterEqual:z:0#cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:i
	cond/CastCastcond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: `
cond/Identity_1Identitycond/Identity:output:0
^cond/NoOp*
T0
*
_output_shapes
: �
	cond/NoOpNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�
�
cond_false_1599292
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
O
	cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 b
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: T
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: "+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: "�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
encoder_input6
serving_default_encoder_input:0���������@E
classifier_output0
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
kernel_quantizer
bias_quantizer
kernel_quantizer_internal
bias_quantizer_internal
 
quantizers

!kernel
"bias"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)
activation
)	quantizer"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0pruning_vars
	1layer
2prunable_weights
3mask
4	threshold
5pruning_step"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<
activation
<	quantizer"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cpruning_vars
	Dlayer
Eprunable_weights
Fmask
G	threshold
Hpruning_step"
_tf_keras_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O
activation
O	quantizer"
_tf_keras_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^pruning_vars
	_layer
`prunable_weights
amask
b	threshold
cpruning_step"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j
activation
j	quantizer"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qkernel_quantizer
rbias_quantizer
qkernel_quantizer_internal
rbias_quantizer_internal
s
quantizers

tkernel
ubias"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|
activation
|	quantizer"
_tf_keras_layer
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�bias_quantizer
�kernel_quantizer_internal
�bias_quantizer_internal
�
quantizers
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
!0
"1
�2
�3
34
45
56
�7
�8
F9
G10
H11
V12
W13
�14
�15
a16
b17
c18
t19
u20
�21
�22"
trackable_list_wrapper
�
!0
"1
�2
�3
�4
�5
V6
W7
�8
�9
t10
u11
�12
�13"
trackable_list_wrapper
0
�0
�1"
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
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
,__inference_classifier_layer_call_fn_1595647
,__inference_classifier_layer_call_fn_1596906
,__inference_classifier_layer_call_fn_1596957
,__inference_classifier_layer_call_fn_1596638�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
G__inference_classifier_layer_call_and_return_conditional_losses_1597608
G__inference_classifier_layer_call_and_return_conditional_losses_1598649
G__inference_classifier_layer_call_and_return_conditional_losses_1596707
G__inference_classifier_layer_call_and_return_conditional_losses_1596788�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
"__inference__wrapped_model_1594825encoder_input"�
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
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
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
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_fc1_layer_call_fn_1598658�
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
 z�trace_0
�
�trace_02�
@__inference_fc1_layer_call_and_return_conditional_losses_1598726�
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
 z�trace_0
"
_generic_user_object
"
_generic_user_object
.
0
1"
trackable_list_wrapper
:@2
fc1/kernel
:2fc1/bias
 "
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
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_relu1_layer_call_fn_1598731�
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
 z�trace_0
�
�trace_02�
B__inference_relu1_layer_call_and_return_conditional_losses_1598780�
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
 z�trace_0
"
_generic_user_object
E
�0
�1
32
43
54"
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
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
>__inference_prune_low_magnitude_fc2_prun_layer_call_fn_1598791
>__inference_prune_low_magnitude_fc2_prun_layer_call_fn_1598806�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1598878
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1599080�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
(
�0"
trackable_list_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�bias_quantizer
�kernel_quantizer_internal
�bias_quantizer_internal
�
quantizers
�kernel
	�bias"
_tf_keras_layer
(
�0"
trackable_list_wrapper
3:1(2!prune_low_magnitude_fc2_prun/mask
0:. (2&prune_low_magnitude_fc2_prun/threshold
1:/	 2)prune_low_magnitude_fc2_prun/pruning_step
 "
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
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_relu2_layer_call_fn_1599085�
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
 z�trace_0
�
�trace_02�
B__inference_relu2_layer_call_and_return_conditional_losses_1599134�
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
 z�trace_0
"
_generic_user_object
E
�0
�1
F2
G3
H4"
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
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
>__inference_prune_low_magnitude_fc3_prun_layer_call_fn_1599145
>__inference_prune_low_magnitude_fc3_prun_layer_call_fn_1599160�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1599232
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1599434�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
(
�0"
trackable_list_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�bias_quantizer
�kernel_quantizer_internal
�bias_quantizer_internal
�
quantizers
�kernel
	�bias"
_tf_keras_layer
(
�0"
trackable_list_wrapper
3:1(2!prune_low_magnitude_fc3_prun/mask
0:. (2&prune_low_magnitude_fc3_prun/threshold
1:/	 2)prune_low_magnitude_fc3_prun/pruning_step
 "
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
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_relu3_enc_layer_call_fn_1599439�
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
 z�trace_0
�
�trace_02�
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1599488�
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
 z�trace_0
"
_generic_user_object
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_encoder_output_layer_call_fn_1599497�
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
 z�trace_0
�
�trace_02�
K__inference_encoder_output_layer_call_and_return_conditional_losses_1599508�
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
 z�trace_0
':%2encoder_output/kernel
!:2encoder_output/bias
E
�0
�1
a2
b3
c4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
E__inference_prune_low_magnitude_fc4_prunedclass_layer_call_fn_1599519
E__inference_prune_low_magnitude_fc4_prunedclass_layer_call_fn_1599534�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1599612
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1599820�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
(
�0"
trackable_list_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�bias_quantizer
�kernel_quantizer_internal
�bias_quantizer_internal
�
quantizers
�kernel
	�bias"
_tf_keras_layer
(
�0"
trackable_list_wrapper
::8(2(prune_low_magnitude_fc4_prunedclass/mask
7:5 (2-prune_low_magnitude_fc4_prunedclass/threshold
8:6	 20prune_low_magnitude_fc4_prunedclass/pruning_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_prunclass_relu4_layer_call_fn_1599831�
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
 z�trace_0
�
�trace_02�
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1599880�
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
 z�trace_0
"
_generic_user_object
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_fc5_class_layer_call_fn_1599889�
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
 z�trace_0
�
�trace_02�
F__inference_fc5_class_layer_call_and_return_conditional_losses_1599963�
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
 z�trace_0
"
_generic_user_object
"
_generic_user_object
.
q0
r1"
trackable_list_wrapper
": (2fc5_class/kernel
:(2fc5_class/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_class_relu5_layer_call_fn_1599968�
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
 z�trace_0
�
�trace_02�
H__inference_class_relu5_layer_call_and_return_conditional_losses_1600017�
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
 z�trace_0
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_classifier_out_layer_call_fn_1600026�
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
 z�trace_0
�
�trace_02�
K__inference_classifier_out_layer_call_and_return_conditional_losses_1600100�
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
 z�trace_0
"
_generic_user_object
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
':%(
2classifier_out/kernel
!:
2classifier_out/bias
 "
trackable_list_wrapper
 "
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
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_classifier_output_layer_call_fn_1600105�
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
 z�trace_0
�
�trace_02�
N__inference_classifier_output_layer_call_and_return_conditional_losses_1600110�
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
 z�trace_0
5:32#prune_low_magnitude_fc2_prun/kernel
/:-2!prune_low_magnitude_fc2_prun/bias
5:32#prune_low_magnitude_fc3_prun/kernel
/:-2!prune_low_magnitude_fc3_prun/bias
<::2*prune_low_magnitude_fc4_prunedclass/kernel
6:42(prune_low_magnitude_fc4_prunedclass/bias
�
�trace_02�
__inference_loss_fn_0_1600121�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_1600132�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
_
30
41
52
F3
G4
H5
a6
b7
c8"
trackable_list_wrapper
�
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
13"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_classifier_layer_call_fn_1595647encoder_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_classifier_layer_call_fn_1596906inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_classifier_layer_call_fn_1596957inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_classifier_layer_call_fn_1596638encoder_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_classifier_layer_call_and_return_conditional_losses_1597608inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_classifier_layer_call_and_return_conditional_losses_1598649inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_classifier_layer_call_and_return_conditional_losses_1596707encoder_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_classifier_layer_call_and_return_conditional_losses_1596788encoder_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�2��
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
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
 0
�B�
%__inference_signature_wrapper_1596849encoder_input"�
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
�B�
%__inference_fc1_layer_call_fn_1598658inputs"�
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
�B�
@__inference_fc1_layer_call_and_return_conditional_losses_1598726inputs"�
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
�B�
'__inference_relu1_layer_call_fn_1598731inputs"�
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
�B�
B__inference_relu1_layer_call_and_return_conditional_losses_1598780inputs"�
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
5
30
41
52"
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
>__inference_prune_low_magnitude_fc2_prun_layer_call_fn_1598791inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_prune_low_magnitude_fc2_prun_layer_call_fn_1598806inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1598878inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1599080inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
7
�0
31
42"
trackable_tuple_wrapper
0
�0
�1"
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
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
"
_generic_user_object
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
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
�B�
'__inference_relu2_layer_call_fn_1599085inputs"�
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
�B�
B__inference_relu2_layer_call_and_return_conditional_losses_1599134inputs"�
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
5
F0
G1
H2"
trackable_list_wrapper
'
D0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
>__inference_prune_low_magnitude_fc3_prun_layer_call_fn_1599145inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_prune_low_magnitude_fc3_prun_layer_call_fn_1599160inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1599232inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1599434inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
7
�0
F1
G2"
trackable_tuple_wrapper
0
�0
�1"
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
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
"
_generic_user_object
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
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
�B�
+__inference_relu3_enc_layer_call_fn_1599439inputs"�
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
�B�
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1599488inputs"�
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
�B�
0__inference_encoder_output_layer_call_fn_1599497inputs"�
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
�B�
K__inference_encoder_output_layer_call_and_return_conditional_losses_1599508inputs"�
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
�
�trace_02�
__inference_loss_fn_2_1600143�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
5
a0
b1
c2"
trackable_list_wrapper
'
_0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
E__inference_prune_low_magnitude_fc4_prunedclass_layer_call_fn_1599519inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_prune_low_magnitude_fc4_prunedclass_layer_call_fn_1599534inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1599612inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1599820inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
7
�0
a1
b2"
trackable_tuple_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
"
_generic_user_object
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
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
�B�
1__inference_prunclass_relu4_layer_call_fn_1599831inputs"�
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
�B�
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1599880inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_fc5_class_layer_call_fn_1599889inputs"�
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
�B�
F__inference_fc5_class_layer_call_and_return_conditional_losses_1599963inputs"�
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
�B�
-__inference_class_relu5_layer_call_fn_1599968inputs"�
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
�B�
H__inference_class_relu5_layer_call_and_return_conditional_losses_1600017inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_classifier_out_layer_call_fn_1600026inputs"�
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
�B�
K__inference_classifier_out_layer_call_and_return_conditional_losses_1600100inputs"�
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
�B�
3__inference_classifier_output_layer_call_fn_1600105inputs"�
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
�B�
N__inference_classifier_output_layer_call_and_return_conditional_losses_1600110inputs"�
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
�B�
__inference_loss_fn_0_1600121"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_1600132"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
!:@2Adam/m/fc1/kernel
!:@2Adam/v/fc1/kernel
:2Adam/m/fc1/bias
:2Adam/v/fc1/bias
::82*Adam/m/prune_low_magnitude_fc2_prun/kernel
::82*Adam/v/prune_low_magnitude_fc2_prun/kernel
4:22(Adam/m/prune_low_magnitude_fc2_prun/bias
4:22(Adam/v/prune_low_magnitude_fc2_prun/bias
::82*Adam/m/prune_low_magnitude_fc3_prun/kernel
::82*Adam/v/prune_low_magnitude_fc3_prun/kernel
4:22(Adam/m/prune_low_magnitude_fc3_prun/bias
4:22(Adam/v/prune_low_magnitude_fc3_prun/bias
,:*2Adam/m/encoder_output/kernel
,:*2Adam/v/encoder_output/kernel
&:$2Adam/m/encoder_output/bias
&:$2Adam/v/encoder_output/bias
A:?21Adam/m/prune_low_magnitude_fc4_prunedclass/kernel
A:?21Adam/v/prune_low_magnitude_fc4_prunedclass/kernel
;:92/Adam/m/prune_low_magnitude_fc4_prunedclass/bias
;:92/Adam/v/prune_low_magnitude_fc4_prunedclass/bias
':%(2Adam/m/fc5_class/kernel
':%(2Adam/v/fc5_class/kernel
!:(2Adam/m/fc5_class/bias
!:(2Adam/v/fc5_class/bias
,:*(
2Adam/m/classifier_out/kernel
,:*(
2Adam/v/classifier_out/kernel
&:$
2Adam/m/classifier_out/bias
&:$
2Adam/v/classifier_out/bias
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
�B�
__inference_loss_fn_2_1600143"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__wrapped_model_1594825�!"�3��F�VW�a�tu��6�3
,�)
'�$
encoder_input���������@
� "E�B
@
classifier_output+�(
classifier_output���������
�
H__inference_class_relu5_layer_call_and_return_conditional_losses_1600017_/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������(
� �
-__inference_class_relu5_layer_call_fn_1599968T/�,
%�"
 �
inputs���������(
� "!�
unknown���������(�
G__inference_classifier_layer_call_and_return_conditional_losses_1596707�!"�3��F�VW�a�tu��>�;
4�1
'�$
encoder_input���������@
p 

 
� ",�)
"�
tensor_0���������

� �
G__inference_classifier_layer_call_and_return_conditional_losses_1596788�!"5�34�H�FG�VWc�ab�tu��>�;
4�1
'�$
encoder_input���������@
p

 
� ",�)
"�
tensor_0���������

� �
G__inference_classifier_layer_call_and_return_conditional_losses_1597608�!"�3��F�VW�a�tu��7�4
-�*
 �
inputs���������@
p 

 
� ",�)
"�
tensor_0���������

� �
G__inference_classifier_layer_call_and_return_conditional_losses_1598649�!"5�34�H�FG�VWc�ab�tu��7�4
-�*
 �
inputs���������@
p

 
� ",�)
"�
tensor_0���������

� �
,__inference_classifier_layer_call_fn_1595647~!"�3��F�VW�a�tu��>�;
4�1
'�$
encoder_input���������@
p 

 
� "!�
unknown���������
�
,__inference_classifier_layer_call_fn_1596638�!"5�34�H�FG�VWc�ab�tu��>�;
4�1
'�$
encoder_input���������@
p

 
� "!�
unknown���������
�
,__inference_classifier_layer_call_fn_1596906w!"�3��F�VW�a�tu��7�4
-�*
 �
inputs���������@
p 

 
� "!�
unknown���������
�
,__inference_classifier_layer_call_fn_1596957}!"5�34�H�FG�VWc�ab�tu��7�4
-�*
 �
inputs���������@
p

 
� "!�
unknown���������
�
K__inference_classifier_out_layer_call_and_return_conditional_losses_1600100e��/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������

� �
0__inference_classifier_out_layer_call_fn_1600026Z��/�,
%�"
 �
inputs���������(
� "!�
unknown���������
�
N__inference_classifier_output_layer_call_and_return_conditional_losses_1600110_/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������

� �
3__inference_classifier_output_layer_call_fn_1600105T/�,
%�"
 �
inputs���������

� "!�
unknown���������
�
K__inference_encoder_output_layer_call_and_return_conditional_losses_1599508cVW/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
0__inference_encoder_output_layer_call_fn_1599497XVW/�,
%�"
 �
inputs���������
� "!�
unknown����������
@__inference_fc1_layer_call_and_return_conditional_losses_1598726c!"/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
%__inference_fc1_layer_call_fn_1598658X!"/�,
%�"
 �
inputs���������@
� "!�
unknown����������
F__inference_fc5_class_layer_call_and_return_conditional_losses_1599963ctu/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������(
� �
+__inference_fc5_class_layer_call_fn_1599889Xtu/�,
%�"
 �
inputs���������
� "!�
unknown���������(E
__inference_loss_fn_0_1600121$t�

� 
� "�
unknown F
__inference_loss_fn_1_1600132%��

� 
� "�
unknown F
__inference_loss_fn_2_1600143%��

� 
� "�
unknown �
L__inference_prunclass_relu4_layer_call_and_return_conditional_losses_1599880_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
1__inference_prunclass_relu4_layer_call_fn_1599831T/�,
%�"
 �
inputs���������
� "!�
unknown����������
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1598878j�3�3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Y__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_1599080l5�34�3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
>__inference_prune_low_magnitude_fc2_prun_layer_call_fn_1598791_�3�3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
>__inference_prune_low_magnitude_fc2_prun_layer_call_fn_1598806a5�34�3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1599232j�F�3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Y__inference_prune_low_magnitude_fc3_prun_layer_call_and_return_conditional_losses_1599434lH�FG�3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
>__inference_prune_low_magnitude_fc3_prun_layer_call_fn_1599145_�F�3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
>__inference_prune_low_magnitude_fc3_prun_layer_call_fn_1599160aH�FG�3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1599612j�a�3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
`__inference_prune_low_magnitude_fc4_prunedclass_layer_call_and_return_conditional_losses_1599820lc�ab�3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
E__inference_prune_low_magnitude_fc4_prunedclass_layer_call_fn_1599519_�a�3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
E__inference_prune_low_magnitude_fc4_prunedclass_layer_call_fn_1599534ac�ab�3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
B__inference_relu1_layer_call_and_return_conditional_losses_1598780_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� 
'__inference_relu1_layer_call_fn_1598731T/�,
%�"
 �
inputs���������
� "!�
unknown����������
B__inference_relu2_layer_call_and_return_conditional_losses_1599134_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� 
'__inference_relu2_layer_call_fn_1599085T/�,
%�"
 �
inputs���������
� "!�
unknown����������
F__inference_relu3_enc_layer_call_and_return_conditional_losses_1599488_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
+__inference_relu3_enc_layer_call_fn_1599439T/�,
%�"
 �
inputs���������
� "!�
unknown����������
%__inference_signature_wrapper_1596849�!"�3��F�VW�a�tu��G�D
� 
=�:
8
encoder_input'�$
encoder_input���������@"E�B
@
classifier_output+�(
classifier_output���������
