��'
��
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
8
Const
output"dtype"
valuetensor"
dtypetype
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
 �"serve*2.11.12v2.11.0-94-ga3e2c692c188��$
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
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
�
Adam/v/classifier_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_nameAdam/v/classifier_output/bias
�
1Adam/v/classifier_output/bias/Read/ReadVariableOpReadVariableOpAdam/v/classifier_output/bias*
_output_shapes
:
*
dtype0
�
Adam/m/classifier_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_nameAdam/m/classifier_output/bias
�
1Adam/m/classifier_output/bias/Read/ReadVariableOpReadVariableOpAdam/m/classifier_output/bias*
_output_shapes
:
*
dtype0
�
Adam/v/classifier_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*0
shared_name!Adam/v/classifier_output/kernel
�
3Adam/v/classifier_output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/classifier_output/kernel*
_output_shapes

:(
*
dtype0
�
Adam/m/classifier_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*0
shared_name!Adam/m/classifier_output/kernel
�
3Adam/m/classifier_output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/classifier_output/kernel*
_output_shapes

:(
*
dtype0
�
Adam/v/ecoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/v/ecoder_output/bias
�
-Adam/v/ecoder_output/bias/Read/ReadVariableOpReadVariableOpAdam/v/ecoder_output/bias*
_output_shapes
:@*
dtype0
�
Adam/m/ecoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/m/ecoder_output/bias
�
-Adam/m/ecoder_output/bias/Read/ReadVariableOpReadVariableOpAdam/m/ecoder_output/bias*
_output_shapes
:@*
dtype0
�
Adam/v/ecoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*,
shared_nameAdam/v/ecoder_output/kernel
�
/Adam/v/ecoder_output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/ecoder_output/kernel*
_output_shapes

: @*
dtype0
�
Adam/m/ecoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*,
shared_nameAdam/m/ecoder_output/kernel
�
/Adam/m/ecoder_output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/ecoder_output/kernel*
_output_shapes

: @*
dtype0
�
Adam/v/fc4_class/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/v/fc4_class/bias
{
)Adam/v/fc4_class/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc4_class/bias*
_output_shapes
:(*
dtype0
�
Adam/m/fc4_class/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/m/fc4_class/bias
{
)Adam/m/fc4_class/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc4_class/bias*
_output_shapes
:(*
dtype0
�
Adam/v/fc4_class/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/v/fc4_class/kernel
�
+Adam/v/fc4_class/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc4_class/kernel*
_output_shapes

:(*
dtype0
�
Adam/m/fc4_class/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*(
shared_nameAdam/m/fc4_class/kernel
�
+Adam/m/fc4_class/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc4_class/kernel*
_output_shapes

:(*
dtype0
z
Adam/v/fc4.1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/fc4.1/bias
s
%Adam/v/fc4.1/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc4.1/bias*
_output_shapes
: *
dtype0
z
Adam/m/fc4.1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/fc4.1/bias
s
%Adam/m/fc4.1/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc4.1/bias*
_output_shapes
: *
dtype0
�
Adam/v/fc4.1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_nameAdam/v/fc4.1/kernel
{
'Adam/v/fc4.1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc4.1/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/fc4.1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_nameAdam/m/fc4.1/kernel
{
'Adam/m/fc4.1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc4.1/kernel*
_output_shapes

:  *
dtype0
�
-Adam/v/prune_low_magnitude_fc3_prunclass/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/v/prune_low_magnitude_fc3_prunclass/bias
�
AAdam/v/prune_low_magnitude_fc3_prunclass/bias/Read/ReadVariableOpReadVariableOp-Adam/v/prune_low_magnitude_fc3_prunclass/bias*
_output_shapes
:*
dtype0
�
-Adam/m/prune_low_magnitude_fc3_prunclass/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/m/prune_low_magnitude_fc3_prunclass/bias
�
AAdam/m/prune_low_magnitude_fc3_prunclass/bias/Read/ReadVariableOpReadVariableOp-Adam/m/prune_low_magnitude_fc3_prunclass/bias*
_output_shapes
:*
dtype0
�
/Adam/v/prune_low_magnitude_fc3_prunclass/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/v/prune_low_magnitude_fc3_prunclass/kernel
�
CAdam/v/prune_low_magnitude_fc3_prunclass/kernel/Read/ReadVariableOpReadVariableOp/Adam/v/prune_low_magnitude_fc3_prunclass/kernel*
_output_shapes

:*
dtype0
�
/Adam/m/prune_low_magnitude_fc3_prunclass/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/Adam/m/prune_low_magnitude_fc3_prunclass/kernel
�
CAdam/m/prune_low_magnitude_fc3_prunclass/kernel/Read/ReadVariableOpReadVariableOp/Adam/m/prune_low_magnitude_fc3_prunclass/kernel*
_output_shapes

:*
dtype0
v
Adam/v/fc4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/v/fc4/bias
o
#Adam/v/fc4/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc4/bias*
_output_shapes
: *
dtype0
v
Adam/m/fc4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/m/fc4/bias
o
#Adam/m/fc4/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc4/bias*
_output_shapes
: *
dtype0
~
Adam/v/fc4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameAdam/v/fc4/kernel
w
%Adam/v/fc4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc4/kernel*
_output_shapes

: *
dtype0
~
Adam/m/fc4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameAdam/m/fc4/kernel
w
%Adam/m/fc4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc4/kernel*
_output_shapes

: *
dtype0
�
*Adam/v/prune_low_magnitude_fc3_pruned/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/v/prune_low_magnitude_fc3_pruned/bias
�
>Adam/v/prune_low_magnitude_fc3_pruned/bias/Read/ReadVariableOpReadVariableOp*Adam/v/prune_low_magnitude_fc3_pruned/bias*
_output_shapes
:*
dtype0
�
*Adam/m/prune_low_magnitude_fc3_pruned/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/m/prune_low_magnitude_fc3_pruned/bias
�
>Adam/m/prune_low_magnitude_fc3_pruned/bias/Read/ReadVariableOpReadVariableOp*Adam/m/prune_low_magnitude_fc3_pruned/bias*
_output_shapes
:*
dtype0
�
,Adam/v/prune_low_magnitude_fc3_pruned/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/v/prune_low_magnitude_fc3_pruned/kernel
�
@Adam/v/prune_low_magnitude_fc3_pruned/kernel/Read/ReadVariableOpReadVariableOp,Adam/v/prune_low_magnitude_fc3_pruned/kernel*
_output_shapes

:*
dtype0
�
,Adam/m/prune_low_magnitude_fc3_pruned/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/m/prune_low_magnitude_fc3_pruned/kernel
�
@Adam/m/prune_low_magnitude_fc3_pruned/kernel/Read/ReadVariableOpReadVariableOp,Adam/m/prune_low_magnitude_fc3_pruned/kernel*
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
*Adam/v/prune_low_magnitude_fc2.1_prun/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/v/prune_low_magnitude_fc2.1_prun/bias
�
>Adam/v/prune_low_magnitude_fc2.1_prun/bias/Read/ReadVariableOpReadVariableOp*Adam/v/prune_low_magnitude_fc2.1_prun/bias*
_output_shapes
:*
dtype0
�
*Adam/m/prune_low_magnitude_fc2.1_prun/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/m/prune_low_magnitude_fc2.1_prun/bias
�
>Adam/m/prune_low_magnitude_fc2.1_prun/bias/Read/ReadVariableOpReadVariableOp*Adam/m/prune_low_magnitude_fc2.1_prun/bias*
_output_shapes
:*
dtype0
�
,Adam/v/prune_low_magnitude_fc2.1_prun/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/v/prune_low_magnitude_fc2.1_prun/kernel
�
@Adam/v/prune_low_magnitude_fc2.1_prun/kernel/Read/ReadVariableOpReadVariableOp,Adam/v/prune_low_magnitude_fc2.1_prun/kernel*
_output_shapes

:*
dtype0
�
,Adam/m/prune_low_magnitude_fc2.1_prun/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/m/prune_low_magnitude_fc2.1_prun/kernel
�
@Adam/m/prune_low_magnitude_fc2.1_prun/kernel/Read/ReadVariableOpReadVariableOp,Adam/m/prune_low_magnitude_fc2.1_prun/kernel*
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
&prune_low_magnitude_fc3_prunclass/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&prune_low_magnitude_fc3_prunclass/bias
�
:prune_low_magnitude_fc3_prunclass/bias/Read/ReadVariableOpReadVariableOp&prune_low_magnitude_fc3_prunclass/bias*
_output_shapes
:*
dtype0
�
(prune_low_magnitude_fc3_prunclass/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(prune_low_magnitude_fc3_prunclass/kernel
�
<prune_low_magnitude_fc3_prunclass/kernel/Read/ReadVariableOpReadVariableOp(prune_low_magnitude_fc3_prunclass/kernel*
_output_shapes

:*
dtype0
�
#prune_low_magnitude_fc3_pruned/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#prune_low_magnitude_fc3_pruned/bias
�
7prune_low_magnitude_fc3_pruned/bias/Read/ReadVariableOpReadVariableOp#prune_low_magnitude_fc3_pruned/bias*
_output_shapes
:*
dtype0
�
%prune_low_magnitude_fc3_pruned/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%prune_low_magnitude_fc3_pruned/kernel
�
9prune_low_magnitude_fc3_pruned/kernel/Read/ReadVariableOpReadVariableOp%prune_low_magnitude_fc3_pruned/kernel*
_output_shapes

:*
dtype0
�
#prune_low_magnitude_fc2.1_prun/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#prune_low_magnitude_fc2.1_prun/bias
�
7prune_low_magnitude_fc2.1_prun/bias/Read/ReadVariableOpReadVariableOp#prune_low_magnitude_fc2.1_prun/bias*
_output_shapes
:*
dtype0
�
%prune_low_magnitude_fc2.1_prun/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%prune_low_magnitude_fc2.1_prun/kernel
�
9prune_low_magnitude_fc2.1_prun/kernel/Read/ReadVariableOpReadVariableOp%prune_low_magnitude_fc2.1_prun/kernel*
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
�
classifier_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameclassifier_output/bias
}
*classifier_output/bias/Read/ReadVariableOpReadVariableOpclassifier_output/bias*
_output_shapes
:
*
dtype0
�
classifier_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*)
shared_nameclassifier_output/kernel
�
,classifier_output/kernel/Read/ReadVariableOpReadVariableOpclassifier_output/kernel*
_output_shapes

:(
*
dtype0
|
ecoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameecoder_output/bias
u
&ecoder_output/bias/Read/ReadVariableOpReadVariableOpecoder_output/bias*
_output_shapes
:@*
dtype0
�
ecoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*%
shared_nameecoder_output/kernel
}
(ecoder_output/kernel/Read/ReadVariableOpReadVariableOpecoder_output/kernel*
_output_shapes

: @*
dtype0
t
fc4_class/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namefc4_class/bias
m
"fc4_class/bias/Read/ReadVariableOpReadVariableOpfc4_class/bias*
_output_shapes
:(*
dtype0
|
fc4_class/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*!
shared_namefc4_class/kernel
u
$fc4_class/kernel/Read/ReadVariableOpReadVariableOpfc4_class/kernel*
_output_shapes

:(*
dtype0
l

fc4.1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
fc4.1/bias
e
fc4.1/bias/Read/ReadVariableOpReadVariableOp
fc4.1/bias*
_output_shapes
: *
dtype0
t
fc4.1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namefc4.1/kernel
m
 fc4.1/kernel/Read/ReadVariableOpReadVariableOpfc4.1/kernel*
_output_shapes

:  *
dtype0
�
.prune_low_magnitude_fc3_prunclass/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *?
shared_name0.prune_low_magnitude_fc3_prunclass/pruning_step
�
Bprune_low_magnitude_fc3_prunclass/pruning_step/Read/ReadVariableOpReadVariableOp.prune_low_magnitude_fc3_prunclass/pruning_step*
_output_shapes
: *
dtype0	
�
+prune_low_magnitude_fc3_prunclass/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+prune_low_magnitude_fc3_prunclass/threshold
�
?prune_low_magnitude_fc3_prunclass/threshold/Read/ReadVariableOpReadVariableOp+prune_low_magnitude_fc3_prunclass/threshold*
_output_shapes
: *
dtype0
�
&prune_low_magnitude_fc3_prunclass/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&prune_low_magnitude_fc3_prunclass/mask
�
:prune_low_magnitude_fc3_prunclass/mask/Read/ReadVariableOpReadVariableOp&prune_low_magnitude_fc3_prunclass/mask*
_output_shapes

:*
dtype0
h
fc4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
fc4/bias
a
fc4/bias/Read/ReadVariableOpReadVariableOpfc4/bias*
_output_shapes
: *
dtype0
p

fc4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_name
fc4/kernel
i
fc4/kernel/Read/ReadVariableOpReadVariableOp
fc4/kernel*
_output_shapes

: *
dtype0
�
+prune_low_magnitude_fc3_pruned/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *<
shared_name-+prune_low_magnitude_fc3_pruned/pruning_step
�
?prune_low_magnitude_fc3_pruned/pruning_step/Read/ReadVariableOpReadVariableOp+prune_low_magnitude_fc3_pruned/pruning_step*
_output_shapes
: *
dtype0	
�
(prune_low_magnitude_fc3_pruned/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(prune_low_magnitude_fc3_pruned/threshold
�
<prune_low_magnitude_fc3_pruned/threshold/Read/ReadVariableOpReadVariableOp(prune_low_magnitude_fc3_pruned/threshold*
_output_shapes
: *
dtype0
�
#prune_low_magnitude_fc3_pruned/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#prune_low_magnitude_fc3_pruned/mask
�
7prune_low_magnitude_fc3_pruned/mask/Read/ReadVariableOpReadVariableOp#prune_low_magnitude_fc3_pruned/mask*
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
+prune_low_magnitude_fc2.1_prun/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *<
shared_name-+prune_low_magnitude_fc2.1_prun/pruning_step
�
?prune_low_magnitude_fc2.1_prun/pruning_step/Read/ReadVariableOpReadVariableOp+prune_low_magnitude_fc2.1_prun/pruning_step*
_output_shapes
: *
dtype0	
�
(prune_low_magnitude_fc2.1_prun/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(prune_low_magnitude_fc2.1_prun/threshold
�
<prune_low_magnitude_fc2.1_prun/threshold/Read/ReadVariableOpReadVariableOp(prune_low_magnitude_fc2.1_prun/threshold*
_output_shapes
: *
dtype0
�
#prune_low_magnitude_fc2.1_prun/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#prune_low_magnitude_fc2.1_prun/mask
�
7prune_low_magnitude_fc2.1_prun/mask/Read/ReadVariableOpReadVariableOp#prune_low_magnitude_fc2.1_prun/mask*
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
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_input
fc1/kernelfc1/bias#prune_low_magnitude_fc2_prun/kernel!prune_low_magnitude_fc2_prun/mask!prune_low_magnitude_fc2_prun/bias%prune_low_magnitude_fc2.1_prun/kernel#prune_low_magnitude_fc2.1_prun/mask#prune_low_magnitude_fc2.1_prun/biasencoder_output/kernelencoder_output/bias%prune_low_magnitude_fc3_pruned/kernel#prune_low_magnitude_fc3_pruned/mask#prune_low_magnitude_fc3_pruned/bias(prune_low_magnitude_fc3_prunclass/kernel&prune_low_magnitude_fc3_prunclass/mask&prune_low_magnitude_fc3_prunclass/bias
fc4/kernelfc4/biasfc4_class/kernelfc4_class/biasfc4.1/kernel
fc4.1/biasclassifier_output/kernelclassifier_output/biasecoder_output/kernelecoder_output/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������@*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_494676

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%pruning_vars
	&layer
'prunable_weights
(mask
)	threshold
*pruning_step*
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1pruning_vars
	2layer
3prunable_weights
4mask
5	threshold
6pruning_step*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Epruning_vars
	Flayer
Gprunable_weights
Hmask
I	threshold
Jpruning_step*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Ypruning_vars
	Zlayer
[prunable_weights
\mask
]	threshold
^pruning_step*
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias*
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias*
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias*
�
0
1
2
�3
(4
)5
*6
�7
�8
49
510
611
=12
>13
�14
�15
H16
I17
J18
Q19
R20
�21
�22
\23
]24
^25
e26
f27
m28
n29
u30
v31
}32
~33*
�
0
1
2
�3
�4
�5
=6
>7
�8
�9
Q10
R11
�12
�13
e14
f15
m16
n17
u18
v19
}20
~21*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
* 

�serving_default* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ZT
VARIABLE_VALUE
fc1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfc1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
(
0
�1
(2
)3
*4*

0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�0*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
	�bias*

0*
oi
VARIABLE_VALUE!prune_low_magnitude_fc2_prun/mask4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE&prune_low_magnitude_fc2_prun/threshold9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE)prune_low_magnitude_fc2_prun/pruning_step<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUE*
)
�0
�1
42
53
64*
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
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�0*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*

�0*
qk
VARIABLE_VALUE#prune_low_magnitude_fc2.1_prun/mask4layer_with_weights-2/mask/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE(prune_low_magnitude_fc2.1_prun/threshold9layer_with_weights-2/threshold/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE+prune_low_magnitude_fc2.1_prun/pruning_step<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUE*

=0
>1*

=0
>1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
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
H2
I3
J4*
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
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�0*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*

�0*
qk
VARIABLE_VALUE#prune_low_magnitude_fc3_pruned/mask4layer_with_weights-4/mask/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE(prune_low_magnitude_fc3_pruned/threshold9layer_with_weights-4/threshold/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE+prune_low_magnitude_fc3_pruned/pruning_step<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ZT
VARIABLE_VALUE
fc4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfc4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
)
�0
�1
\2
]3
^4*
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
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�0*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*

�0*
tn
VARIABLE_VALUE&prune_low_magnitude_fc3_prunclass/mask4layer_with_weights-6/mask/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE+prune_low_magnitude_fc3_prunclass/threshold9layer_with_weights-6/threshold/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUE.prune_low_magnitude_fc3_prunclass/pruning_step<layer_with_weights-6/pruning_step/.ATTRIBUTES/VARIABLE_VALUE*

e0
f1*

e0
f1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEfc4.1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
fc4.1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

m0
n1*

m0
n1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEfc4_class/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEfc4_class/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

u0
v1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEecoder_output/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEecoder_output/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

}0
~1*

}0
~1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ic
VARIABLE_VALUEclassifier_output/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEclassifier_output/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#prune_low_magnitude_fc2_prun/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!prune_low_magnitude_fc2_prun/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%prune_low_magnitude_fc2.1_prun/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#prune_low_magnitude_fc2.1_prun/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%prune_low_magnitude_fc3_pruned/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#prune_low_magnitude_fc3_pruned/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(prune_low_magnitude_fc3_prunclass/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&prune_low_magnitude_fc3_prunclass/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
Z
(0
)1
*2
43
54
65
H6
I7
J8
\9
]10
^11*
Z
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
11*
,
�0
�1
�2
�3
�4*
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
�
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
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
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
�13
�14
�15
�16
�17
�18
�19
�20
�21*
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
�13
�14
�15
�16
�17
�18
�19
�20
�21*
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
(0
)1
*2*

&0*
* 
* 
* 
* 
* 
* 
* 

0
(1
)2*

0
�1*

0
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

40
51
62*

20*
* 
* 
* 
* 
* 
* 
* 

�0
41
52*
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
* 
* 
* 
* 
* 

H0
I1
J2*

F0*
* 
* 
* 
* 
* 
* 
* 

�0
H1
I2*
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
* 
* 
* 
* 
* 

\0
]1
^2*

Z0*
* 
* 
* 
* 
* 
* 
* 

�0
\1
]2*
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
<
�	variables
�	keras_api

�total

�count*
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
wq
VARIABLE_VALUE,Adam/m/prune_low_magnitude_fc2.1_prun/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,Adam/v/prune_low_magnitude_fc2.1_prun/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/prune_low_magnitude_fc2.1_prun/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/prune_low_magnitude_fc2.1_prun/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/encoder_output/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/encoder_output/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/encoder_output/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/encoder_output/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,Adam/m/prune_low_magnitude_fc3_pruned/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,Adam/v/prune_low_magnitude_fc3_pruned/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/prune_low_magnitude_fc3_pruned/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/prune_low_magnitude_fc3_pruned/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/fc4/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/fc4/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/fc4/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/fc4/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE/Adam/m/prune_low_magnitude_fc3_prunclass/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE/Adam/v/prune_low_magnitude_fc3_prunclass/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/m/prune_low_magnitude_fc3_prunclass/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/v/prune_low_magnitude_fc3_prunclass/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/fc4.1/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/fc4.1/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/fc4.1/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/fc4.1/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/fc4_class/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/fc4_class/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/fc4_class/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/fc4_class/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/ecoder_output/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/ecoder_output/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/ecoder_output/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/ecoder_output/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/classifier_output/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/classifier_output/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/classifier_output/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/classifier_output/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamefc1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOp5prune_low_magnitude_fc2_prun/mask/Read/ReadVariableOp:prune_low_magnitude_fc2_prun/threshold/Read/ReadVariableOp=prune_low_magnitude_fc2_prun/pruning_step/Read/ReadVariableOp7prune_low_magnitude_fc2.1_prun/mask/Read/ReadVariableOp<prune_low_magnitude_fc2.1_prun/threshold/Read/ReadVariableOp?prune_low_magnitude_fc2.1_prun/pruning_step/Read/ReadVariableOp)encoder_output/kernel/Read/ReadVariableOp'encoder_output/bias/Read/ReadVariableOp7prune_low_magnitude_fc3_pruned/mask/Read/ReadVariableOp<prune_low_magnitude_fc3_pruned/threshold/Read/ReadVariableOp?prune_low_magnitude_fc3_pruned/pruning_step/Read/ReadVariableOpfc4/kernel/Read/ReadVariableOpfc4/bias/Read/ReadVariableOp:prune_low_magnitude_fc3_prunclass/mask/Read/ReadVariableOp?prune_low_magnitude_fc3_prunclass/threshold/Read/ReadVariableOpBprune_low_magnitude_fc3_prunclass/pruning_step/Read/ReadVariableOp fc4.1/kernel/Read/ReadVariableOpfc4.1/bias/Read/ReadVariableOp$fc4_class/kernel/Read/ReadVariableOp"fc4_class/bias/Read/ReadVariableOp(ecoder_output/kernel/Read/ReadVariableOp&ecoder_output/bias/Read/ReadVariableOp,classifier_output/kernel/Read/ReadVariableOp*classifier_output/bias/Read/ReadVariableOp7prune_low_magnitude_fc2_prun/kernel/Read/ReadVariableOp5prune_low_magnitude_fc2_prun/bias/Read/ReadVariableOp9prune_low_magnitude_fc2.1_prun/kernel/Read/ReadVariableOp7prune_low_magnitude_fc2.1_prun/bias/Read/ReadVariableOp9prune_low_magnitude_fc3_pruned/kernel/Read/ReadVariableOp7prune_low_magnitude_fc3_pruned/bias/Read/ReadVariableOp<prune_low_magnitude_fc3_prunclass/kernel/Read/ReadVariableOp:prune_low_magnitude_fc3_prunclass/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp%Adam/m/fc1/kernel/Read/ReadVariableOp%Adam/v/fc1/kernel/Read/ReadVariableOp#Adam/m/fc1/bias/Read/ReadVariableOp#Adam/v/fc1/bias/Read/ReadVariableOp>Adam/m/prune_low_magnitude_fc2_prun/kernel/Read/ReadVariableOp>Adam/v/prune_low_magnitude_fc2_prun/kernel/Read/ReadVariableOp<Adam/m/prune_low_magnitude_fc2_prun/bias/Read/ReadVariableOp<Adam/v/prune_low_magnitude_fc2_prun/bias/Read/ReadVariableOp@Adam/m/prune_low_magnitude_fc2.1_prun/kernel/Read/ReadVariableOp@Adam/v/prune_low_magnitude_fc2.1_prun/kernel/Read/ReadVariableOp>Adam/m/prune_low_magnitude_fc2.1_prun/bias/Read/ReadVariableOp>Adam/v/prune_low_magnitude_fc2.1_prun/bias/Read/ReadVariableOp0Adam/m/encoder_output/kernel/Read/ReadVariableOp0Adam/v/encoder_output/kernel/Read/ReadVariableOp.Adam/m/encoder_output/bias/Read/ReadVariableOp.Adam/v/encoder_output/bias/Read/ReadVariableOp@Adam/m/prune_low_magnitude_fc3_pruned/kernel/Read/ReadVariableOp@Adam/v/prune_low_magnitude_fc3_pruned/kernel/Read/ReadVariableOp>Adam/m/prune_low_magnitude_fc3_pruned/bias/Read/ReadVariableOp>Adam/v/prune_low_magnitude_fc3_pruned/bias/Read/ReadVariableOp%Adam/m/fc4/kernel/Read/ReadVariableOp%Adam/v/fc4/kernel/Read/ReadVariableOp#Adam/m/fc4/bias/Read/ReadVariableOp#Adam/v/fc4/bias/Read/ReadVariableOpCAdam/m/prune_low_magnitude_fc3_prunclass/kernel/Read/ReadVariableOpCAdam/v/prune_low_magnitude_fc3_prunclass/kernel/Read/ReadVariableOpAAdam/m/prune_low_magnitude_fc3_prunclass/bias/Read/ReadVariableOpAAdam/v/prune_low_magnitude_fc3_prunclass/bias/Read/ReadVariableOp'Adam/m/fc4.1/kernel/Read/ReadVariableOp'Adam/v/fc4.1/kernel/Read/ReadVariableOp%Adam/m/fc4.1/bias/Read/ReadVariableOp%Adam/v/fc4.1/bias/Read/ReadVariableOp+Adam/m/fc4_class/kernel/Read/ReadVariableOp+Adam/v/fc4_class/kernel/Read/ReadVariableOp)Adam/m/fc4_class/bias/Read/ReadVariableOp)Adam/v/fc4_class/bias/Read/ReadVariableOp/Adam/m/ecoder_output/kernel/Read/ReadVariableOp/Adam/v/ecoder_output/kernel/Read/ReadVariableOp-Adam/m/ecoder_output/bias/Read/ReadVariableOp-Adam/v/ecoder_output/bias/Read/ReadVariableOp3Adam/m/classifier_output/kernel/Read/ReadVariableOp3Adam/v/classifier_output/kernel/Read/ReadVariableOp1Adam/m/classifier_output/bias/Read/ReadVariableOp1Adam/v/classifier_output/bias/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*g
Tin`
^2\					*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_496704
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
fc1/kernelfc1/bias!prune_low_magnitude_fc2_prun/mask&prune_low_magnitude_fc2_prun/threshold)prune_low_magnitude_fc2_prun/pruning_step#prune_low_magnitude_fc2.1_prun/mask(prune_low_magnitude_fc2.1_prun/threshold+prune_low_magnitude_fc2.1_prun/pruning_stepencoder_output/kernelencoder_output/bias#prune_low_magnitude_fc3_pruned/mask(prune_low_magnitude_fc3_pruned/threshold+prune_low_magnitude_fc3_pruned/pruning_step
fc4/kernelfc4/bias&prune_low_magnitude_fc3_prunclass/mask+prune_low_magnitude_fc3_prunclass/threshold.prune_low_magnitude_fc3_prunclass/pruning_stepfc4.1/kernel
fc4.1/biasfc4_class/kernelfc4_class/biasecoder_output/kernelecoder_output/biasclassifier_output/kernelclassifier_output/bias#prune_low_magnitude_fc2_prun/kernel!prune_low_magnitude_fc2_prun/bias%prune_low_magnitude_fc2.1_prun/kernel#prune_low_magnitude_fc2.1_prun/bias%prune_low_magnitude_fc3_pruned/kernel#prune_low_magnitude_fc3_pruned/bias(prune_low_magnitude_fc3_prunclass/kernel&prune_low_magnitude_fc3_prunclass/bias	iterationlearning_rateAdam/m/fc1/kernelAdam/v/fc1/kernelAdam/m/fc1/biasAdam/v/fc1/bias*Adam/m/prune_low_magnitude_fc2_prun/kernel*Adam/v/prune_low_magnitude_fc2_prun/kernel(Adam/m/prune_low_magnitude_fc2_prun/bias(Adam/v/prune_low_magnitude_fc2_prun/bias,Adam/m/prune_low_magnitude_fc2.1_prun/kernel,Adam/v/prune_low_magnitude_fc2.1_prun/kernel*Adam/m/prune_low_magnitude_fc2.1_prun/bias*Adam/v/prune_low_magnitude_fc2.1_prun/biasAdam/m/encoder_output/kernelAdam/v/encoder_output/kernelAdam/m/encoder_output/biasAdam/v/encoder_output/bias,Adam/m/prune_low_magnitude_fc3_pruned/kernel,Adam/v/prune_low_magnitude_fc3_pruned/kernel*Adam/m/prune_low_magnitude_fc3_pruned/bias*Adam/v/prune_low_magnitude_fc3_pruned/biasAdam/m/fc4/kernelAdam/v/fc4/kernelAdam/m/fc4/biasAdam/v/fc4/bias/Adam/m/prune_low_magnitude_fc3_prunclass/kernel/Adam/v/prune_low_magnitude_fc3_prunclass/kernel-Adam/m/prune_low_magnitude_fc3_prunclass/bias-Adam/v/prune_low_magnitude_fc3_prunclass/biasAdam/m/fc4.1/kernelAdam/v/fc4.1/kernelAdam/m/fc4.1/biasAdam/v/fc4.1/biasAdam/m/fc4_class/kernelAdam/v/fc4_class/kernelAdam/m/fc4_class/biasAdam/v/fc4_class/biasAdam/m/ecoder_output/kernelAdam/v/ecoder_output/kernelAdam/m/ecoder_output/biasAdam/v/ecoder_output/biasAdam/m/classifier_output/kernelAdam/v/classifier_output/kernelAdam/m/classifier_output/biasAdam/v/classifier_output/biastotal_4count_4total_3count_3total_2count_2total_1count_1totalcount*f
Tin_
]2[*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_496984�� 
�
�
*__inference_fc4_class_layer_call_fn_496359

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
GPU 2J 8� *N
fIRG
E__inference_fc4_class_layer_call_and_return_conditional_losses_493294o
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
�
�
3assert_greater_equal_Assert_AssertGuard_true_493493M
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
3assert_greater_equal_Assert_AssertGuard_true_493665M
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
�
?__inference_prune_low_magnitude_fc3_pruned_layer_call_fn_495964

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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_493791o
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
�
3assert_greater_equal_Assert_AssertGuard_true_495792M
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
ۦ
�*
__inference__traced_save_496704
file_prefix)
%savev2_fc1_kernel_read_readvariableop'
#savev2_fc1_bias_read_readvariableop@
<savev2_prune_low_magnitude_fc2_prun_mask_read_readvariableopE
Asavev2_prune_low_magnitude_fc2_prun_threshold_read_readvariableopH
Dsavev2_prune_low_magnitude_fc2_prun_pruning_step_read_readvariableop	B
>savev2_prune_low_magnitude_fc2_1_prun_mask_read_readvariableopG
Csavev2_prune_low_magnitude_fc2_1_prun_threshold_read_readvariableopJ
Fsavev2_prune_low_magnitude_fc2_1_prun_pruning_step_read_readvariableop	4
0savev2_encoder_output_kernel_read_readvariableop2
.savev2_encoder_output_bias_read_readvariableopB
>savev2_prune_low_magnitude_fc3_pruned_mask_read_readvariableopG
Csavev2_prune_low_magnitude_fc3_pruned_threshold_read_readvariableopJ
Fsavev2_prune_low_magnitude_fc3_pruned_pruning_step_read_readvariableop	)
%savev2_fc4_kernel_read_readvariableop'
#savev2_fc4_bias_read_readvariableopE
Asavev2_prune_low_magnitude_fc3_prunclass_mask_read_readvariableopJ
Fsavev2_prune_low_magnitude_fc3_prunclass_threshold_read_readvariableopM
Isavev2_prune_low_magnitude_fc3_prunclass_pruning_step_read_readvariableop	+
'savev2_fc4_1_kernel_read_readvariableop)
%savev2_fc4_1_bias_read_readvariableop/
+savev2_fc4_class_kernel_read_readvariableop-
)savev2_fc4_class_bias_read_readvariableop3
/savev2_ecoder_output_kernel_read_readvariableop1
-savev2_ecoder_output_bias_read_readvariableop7
3savev2_classifier_output_kernel_read_readvariableop5
1savev2_classifier_output_bias_read_readvariableopB
>savev2_prune_low_magnitude_fc2_prun_kernel_read_readvariableop@
<savev2_prune_low_magnitude_fc2_prun_bias_read_readvariableopD
@savev2_prune_low_magnitude_fc2_1_prun_kernel_read_readvariableopB
>savev2_prune_low_magnitude_fc2_1_prun_bias_read_readvariableopD
@savev2_prune_low_magnitude_fc3_pruned_kernel_read_readvariableopB
>savev2_prune_low_magnitude_fc3_pruned_bias_read_readvariableopG
Csavev2_prune_low_magnitude_fc3_prunclass_kernel_read_readvariableopE
Asavev2_prune_low_magnitude_fc3_prunclass_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop0
,savev2_adam_m_fc1_kernel_read_readvariableop0
,savev2_adam_v_fc1_kernel_read_readvariableop.
*savev2_adam_m_fc1_bias_read_readvariableop.
*savev2_adam_v_fc1_bias_read_readvariableopI
Esavev2_adam_m_prune_low_magnitude_fc2_prun_kernel_read_readvariableopI
Esavev2_adam_v_prune_low_magnitude_fc2_prun_kernel_read_readvariableopG
Csavev2_adam_m_prune_low_magnitude_fc2_prun_bias_read_readvariableopG
Csavev2_adam_v_prune_low_magnitude_fc2_prun_bias_read_readvariableopK
Gsavev2_adam_m_prune_low_magnitude_fc2_1_prun_kernel_read_readvariableopK
Gsavev2_adam_v_prune_low_magnitude_fc2_1_prun_kernel_read_readvariableopI
Esavev2_adam_m_prune_low_magnitude_fc2_1_prun_bias_read_readvariableopI
Esavev2_adam_v_prune_low_magnitude_fc2_1_prun_bias_read_readvariableop;
7savev2_adam_m_encoder_output_kernel_read_readvariableop;
7savev2_adam_v_encoder_output_kernel_read_readvariableop9
5savev2_adam_m_encoder_output_bias_read_readvariableop9
5savev2_adam_v_encoder_output_bias_read_readvariableopK
Gsavev2_adam_m_prune_low_magnitude_fc3_pruned_kernel_read_readvariableopK
Gsavev2_adam_v_prune_low_magnitude_fc3_pruned_kernel_read_readvariableopI
Esavev2_adam_m_prune_low_magnitude_fc3_pruned_bias_read_readvariableopI
Esavev2_adam_v_prune_low_magnitude_fc3_pruned_bias_read_readvariableop0
,savev2_adam_m_fc4_kernel_read_readvariableop0
,savev2_adam_v_fc4_kernel_read_readvariableop.
*savev2_adam_m_fc4_bias_read_readvariableop.
*savev2_adam_v_fc4_bias_read_readvariableopN
Jsavev2_adam_m_prune_low_magnitude_fc3_prunclass_kernel_read_readvariableopN
Jsavev2_adam_v_prune_low_magnitude_fc3_prunclass_kernel_read_readvariableopL
Hsavev2_adam_m_prune_low_magnitude_fc3_prunclass_bias_read_readvariableopL
Hsavev2_adam_v_prune_low_magnitude_fc3_prunclass_bias_read_readvariableop2
.savev2_adam_m_fc4_1_kernel_read_readvariableop2
.savev2_adam_v_fc4_1_kernel_read_readvariableop0
,savev2_adam_m_fc4_1_bias_read_readvariableop0
,savev2_adam_v_fc4_1_bias_read_readvariableop6
2savev2_adam_m_fc4_class_kernel_read_readvariableop6
2savev2_adam_v_fc4_class_kernel_read_readvariableop4
0savev2_adam_m_fc4_class_bias_read_readvariableop4
0savev2_adam_v_fc4_class_bias_read_readvariableop:
6savev2_adam_m_ecoder_output_kernel_read_readvariableop:
6savev2_adam_v_ecoder_output_kernel_read_readvariableop8
4savev2_adam_m_ecoder_output_bias_read_readvariableop8
4savev2_adam_v_ecoder_output_bias_read_readvariableop>
:savev2_adam_m_classifier_output_kernel_read_readvariableop>
:savev2_adam_v_classifier_output_kernel_read_readvariableop<
8savev2_adam_m_classifier_output_bias_read_readvariableop<
8savev2_adam_v_classifier_output_bias_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
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
: �%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*�%
value�%B�%[B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-2/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-4/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-6/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-6/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*�
value�B�[B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop<savev2_prune_low_magnitude_fc2_prun_mask_read_readvariableopAsavev2_prune_low_magnitude_fc2_prun_threshold_read_readvariableopDsavev2_prune_low_magnitude_fc2_prun_pruning_step_read_readvariableop>savev2_prune_low_magnitude_fc2_1_prun_mask_read_readvariableopCsavev2_prune_low_magnitude_fc2_1_prun_threshold_read_readvariableopFsavev2_prune_low_magnitude_fc2_1_prun_pruning_step_read_readvariableop0savev2_encoder_output_kernel_read_readvariableop.savev2_encoder_output_bias_read_readvariableop>savev2_prune_low_magnitude_fc3_pruned_mask_read_readvariableopCsavev2_prune_low_magnitude_fc3_pruned_threshold_read_readvariableopFsavev2_prune_low_magnitude_fc3_pruned_pruning_step_read_readvariableop%savev2_fc4_kernel_read_readvariableop#savev2_fc4_bias_read_readvariableopAsavev2_prune_low_magnitude_fc3_prunclass_mask_read_readvariableopFsavev2_prune_low_magnitude_fc3_prunclass_threshold_read_readvariableopIsavev2_prune_low_magnitude_fc3_prunclass_pruning_step_read_readvariableop'savev2_fc4_1_kernel_read_readvariableop%savev2_fc4_1_bias_read_readvariableop+savev2_fc4_class_kernel_read_readvariableop)savev2_fc4_class_bias_read_readvariableop/savev2_ecoder_output_kernel_read_readvariableop-savev2_ecoder_output_bias_read_readvariableop3savev2_classifier_output_kernel_read_readvariableop1savev2_classifier_output_bias_read_readvariableop>savev2_prune_low_magnitude_fc2_prun_kernel_read_readvariableop<savev2_prune_low_magnitude_fc2_prun_bias_read_readvariableop@savev2_prune_low_magnitude_fc2_1_prun_kernel_read_readvariableop>savev2_prune_low_magnitude_fc2_1_prun_bias_read_readvariableop@savev2_prune_low_magnitude_fc3_pruned_kernel_read_readvariableop>savev2_prune_low_magnitude_fc3_pruned_bias_read_readvariableopCsavev2_prune_low_magnitude_fc3_prunclass_kernel_read_readvariableopAsavev2_prune_low_magnitude_fc3_prunclass_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop,savev2_adam_m_fc1_kernel_read_readvariableop,savev2_adam_v_fc1_kernel_read_readvariableop*savev2_adam_m_fc1_bias_read_readvariableop*savev2_adam_v_fc1_bias_read_readvariableopEsavev2_adam_m_prune_low_magnitude_fc2_prun_kernel_read_readvariableopEsavev2_adam_v_prune_low_magnitude_fc2_prun_kernel_read_readvariableopCsavev2_adam_m_prune_low_magnitude_fc2_prun_bias_read_readvariableopCsavev2_adam_v_prune_low_magnitude_fc2_prun_bias_read_readvariableopGsavev2_adam_m_prune_low_magnitude_fc2_1_prun_kernel_read_readvariableopGsavev2_adam_v_prune_low_magnitude_fc2_1_prun_kernel_read_readvariableopEsavev2_adam_m_prune_low_magnitude_fc2_1_prun_bias_read_readvariableopEsavev2_adam_v_prune_low_magnitude_fc2_1_prun_bias_read_readvariableop7savev2_adam_m_encoder_output_kernel_read_readvariableop7savev2_adam_v_encoder_output_kernel_read_readvariableop5savev2_adam_m_encoder_output_bias_read_readvariableop5savev2_adam_v_encoder_output_bias_read_readvariableopGsavev2_adam_m_prune_low_magnitude_fc3_pruned_kernel_read_readvariableopGsavev2_adam_v_prune_low_magnitude_fc3_pruned_kernel_read_readvariableopEsavev2_adam_m_prune_low_magnitude_fc3_pruned_bias_read_readvariableopEsavev2_adam_v_prune_low_magnitude_fc3_pruned_bias_read_readvariableop,savev2_adam_m_fc4_kernel_read_readvariableop,savev2_adam_v_fc4_kernel_read_readvariableop*savev2_adam_m_fc4_bias_read_readvariableop*savev2_adam_v_fc4_bias_read_readvariableopJsavev2_adam_m_prune_low_magnitude_fc3_prunclass_kernel_read_readvariableopJsavev2_adam_v_prune_low_magnitude_fc3_prunclass_kernel_read_readvariableopHsavev2_adam_m_prune_low_magnitude_fc3_prunclass_bias_read_readvariableopHsavev2_adam_v_prune_low_magnitude_fc3_prunclass_bias_read_readvariableop.savev2_adam_m_fc4_1_kernel_read_readvariableop.savev2_adam_v_fc4_1_kernel_read_readvariableop,savev2_adam_m_fc4_1_bias_read_readvariableop,savev2_adam_v_fc4_1_bias_read_readvariableop2savev2_adam_m_fc4_class_kernel_read_readvariableop2savev2_adam_v_fc4_class_kernel_read_readvariableop0savev2_adam_m_fc4_class_bias_read_readvariableop0savev2_adam_v_fc4_class_bias_read_readvariableop6savev2_adam_m_ecoder_output_kernel_read_readvariableop6savev2_adam_v_ecoder_output_kernel_read_readvariableop4savev2_adam_m_ecoder_output_bias_read_readvariableop4savev2_adam_v_ecoder_output_bias_read_readvariableop:savev2_adam_m_classifier_output_kernel_read_readvariableop:savev2_adam_v_classifier_output_kernel_read_readvariableop8savev2_adam_m_classifier_output_bias_read_readvariableop8savev2_adam_v_classifier_output_bias_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *i
dtypes_
]2[					�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :@::: : :: : :::: : : : :: : :  : :(:(: @:@:(
:
::::::::: : :@:@::::::::::::::::::: : : : :::::  :  : : :(:(:(:(: @: @:@:@:(
:(
:
:
: : : : : : : : : : : 2(
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

: : 

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:(: 

_output_shapes
:(:$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:(
: 

_output_shapes
:
:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::#

_output_shapes
: :$

_output_shapes
: :$% 

_output_shapes

:@:$& 

_output_shapes

:@: '

_output_shapes
:: (

_output_shapes
::$) 

_output_shapes

::$* 

_output_shapes

:: +

_output_shapes
:: ,

_output_shapes
::$- 

_output_shapes

::$. 

_output_shapes

:: /

_output_shapes
:: 0

_output_shapes
::$1 

_output_shapes

::$2 

_output_shapes

:: 3

_output_shapes
:: 4

_output_shapes
::$5 

_output_shapes

::$6 

_output_shapes

:: 7

_output_shapes
:: 8

_output_shapes
::$9 

_output_shapes

: :$: 

_output_shapes

: : ;

_output_shapes
: : <

_output_shapes
: :$= 

_output_shapes

::$> 

_output_shapes

:: ?

_output_shapes
:: @

_output_shapes
::$A 

_output_shapes

:  :$B 

_output_shapes

:  : C

_output_shapes
: : D

_output_shapes
: :$E 

_output_shapes

:(:$F 

_output_shapes

:(: G

_output_shapes
:(: H

_output_shapes
:(:$I 

_output_shapes

: @:$J 

_output_shapes

: @: K

_output_shapes
:@: L

_output_shapes
:@:$M 

_output_shapes

:(
:$N 

_output_shapes

:(
: O

_output_shapes
:
: P

_output_shapes
:
:Q

_output_shapes
: :R

_output_shapes
: :S

_output_shapes
: :T

_output_shapes
: :U

_output_shapes
: :V

_output_shapes
: :W

_output_shapes
: :X

_output_shapes
: :Y

_output_shapes
: :Z

_output_shapes
: :[

_output_shapes
: 
�
�
cond_false_496245
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
�
�
?__inference_prune_low_magnitude_fc2.1_prun_layer_call_fn_495743

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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_493195o
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
�
�
4assert_greater_equal_Assert_AssertGuard_false_495999K
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
$__inference_fc1_layer_call_fn_495535

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
GPU 2J 8� *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_493151o
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
�
�
$__inference_signature_wrapper_494676
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

unknown_12:

unknown_13:

unknown_14:

unknown_15: 

unknown_16: 

unknown_17:(

unknown_18:(

unknown_19:  

unknown_20: 

unknown_21:(


unknown_22:


unknown_23: @

unknown_24:@
identity

identity_1��StatefulPartitionedCall�
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
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������@*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_493133o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�

�
?__inference_fc4_layer_call_and_return_conditional_losses_496144

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_493195

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��AssignVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1'
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
 �
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3assert_greater_equal_Assert_AssertGuard_true_494019M
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
�

�
?__inference_fc4_layer_call_and_return_conditional_losses_493277

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.prune_low_magnitude_fc2_prun_cond_false_4949751
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
�
�
3assert_greater_equal_Assert_AssertGuard_true_496204M
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
�F
�
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_495732

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: -
biasadd_readvariableop_resource:
identity��AssignVariableOp�AssignVariableOp_1�BiasAdd/ReadVariableOp�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
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
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_495607*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_495606�
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
 *��Y?�
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
*$
else_branchR
cond_false_495647*
output_shapes
: *#
then_branchR
cond_true_495646q
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
 w
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_120
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�F
�
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_496330

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: -
biasadd_readvariableop_resource:
identity��AssignVariableOp�AssignVariableOp_1�BiasAdd/ReadVariableOp�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
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
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_496205*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_496204�
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
 *��Y?�
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
*$
else_branchR
cond_false_496245*
output_shapes
: *#
then_branchR
cond_true_496244q
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
 w
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_120
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_model_1_layer_call_fn_494735

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

unknown_12:

unknown_13:

unknown_14:

unknown_15: 

unknown_16: 

unknown_17:(

unknown_18:(

unknown_19:  

unknown_20: 

unknown_21:(


unknown_22:


unknown_23: @

unknown_24:@
identity

identity_1��StatefulPartitionedCall�
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
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������
*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_493353o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
A__inference_fc4.1_layer_call_and_return_conditional_losses_496350

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
$__inference_fc4_layer_call_fn_496133

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_fc4_layer_call_and_return_conditional_losses_493277o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
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
cond_true_4937053
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
 *��Y?�
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

�
J__inference_encoder_output_layer_call_and_return_conditional_losses_495938

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
�i
�
2prune_low_magnitude_fc3_prunclass_cond_true_495404U
Kprune_low_magnitude_fc3_prunclass_cond_greaterequal_readvariableop_resource:	 `
Nprune_low_magnitude_fc3_prunclass_cond_pruning_ops_abs_readvariableop_resource:R
@prune_low_magnitude_fc3_prunclass_cond_assignvariableop_resource:L
Bprune_low_magnitude_fc3_prunclass_cond_assignvariableop_1_resource: b
^prune_low_magnitude_fc3_prunclass_cond_identity_prune_low_magnitude_fc3_prunclass_logicaland_1
5
1prune_low_magnitude_fc3_prunclass_cond_identity_1
��7prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp�9prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp_1�Bprune_low_magnitude_fc3_prunclass/cond/GreaterEqual/ReadVariableOp�?prune_low_magnitude_fc3_prunclass/cond/LessEqual/ReadVariableOp�9prune_low_magnitude_fc3_prunclass/cond/Sub/ReadVariableOp�Eprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Abs/ReadVariableOp�
Bprune_low_magnitude_fc3_prunclass/cond/GreaterEqual/ReadVariableOpReadVariableOpKprune_low_magnitude_fc3_prunclass_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	x
5prune_low_magnitude_fc3_prunclass/cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
3prune_low_magnitude_fc3_prunclass/cond/GreaterEqualGreaterEqualJprune_low_magnitude_fc3_prunclass/cond/GreaterEqual/ReadVariableOp:value:0>prune_low_magnitude_fc3_prunclass/cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
?prune_low_magnitude_fc3_prunclass/cond/LessEqual/ReadVariableOpReadVariableOpKprune_low_magnitude_fc3_prunclass_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	u
2prune_low_magnitude_fc3_prunclass/cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N�
0prune_low_magnitude_fc3_prunclass/cond/LessEqual	LessEqualGprune_low_magnitude_fc3_prunclass/cond/LessEqual/ReadVariableOp:value:0;prune_low_magnitude_fc3_prunclass/cond/LessEqual/y:output:0*
T0	*
_output_shapes
: p
-prune_low_magnitude_fc3_prunclass/cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�No
-prune_low_magnitude_fc3_prunclass/cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : �
+prune_low_magnitude_fc3_prunclass/cond/LessLess6prune_low_magnitude_fc3_prunclass/cond/Less/x:output:06prune_low_magnitude_fc3_prunclass/cond/Less/y:output:0*
T0*
_output_shapes
: �
0prune_low_magnitude_fc3_prunclass/cond/LogicalOr	LogicalOr4prune_low_magnitude_fc3_prunclass/cond/LessEqual:z:0/prune_low_magnitude_fc3_prunclass/cond/Less:z:0*
_output_shapes
: �
1prune_low_magnitude_fc3_prunclass/cond/LogicalAnd
LogicalAnd7prune_low_magnitude_fc3_prunclass/cond/GreaterEqual:z:04prune_low_magnitude_fc3_prunclass/cond/LogicalOr:z:0*
_output_shapes
: �
9prune_low_magnitude_fc3_prunclass/cond/Sub/ReadVariableOpReadVariableOpKprune_low_magnitude_fc3_prunclass_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	o
,prune_low_magnitude_fc3_prunclass/cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
*prune_low_magnitude_fc3_prunclass/cond/SubSubAprune_low_magnitude_fc3_prunclass/cond/Sub/ReadVariableOp:value:05prune_low_magnitude_fc3_prunclass/cond/Sub/y:output:0*
T0	*
_output_shapes
: s
1prune_low_magnitude_fc3_prunclass/cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd�
/prune_low_magnitude_fc3_prunclass/cond/FloorModFloorMod.prune_low_magnitude_fc3_prunclass/cond/Sub:z:0:prune_low_magnitude_fc3_prunclass/cond/FloorMod/y:output:0*
T0	*
_output_shapes
: p
.prune_low_magnitude_fc3_prunclass/cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
,prune_low_magnitude_fc3_prunclass/cond/EqualEqual3prune_low_magnitude_fc3_prunclass/cond/FloorMod:z:07prune_low_magnitude_fc3_prunclass/cond/Equal/y:output:0*
T0	*
_output_shapes
: �
3prune_low_magnitude_fc3_prunclass/cond/LogicalAnd_1
LogicalAnd5prune_low_magnitude_fc3_prunclass/cond/LogicalAnd:z:00prune_low_magnitude_fc3_prunclass/cond/Equal:z:0*
_output_shapes
: q
,prune_low_magnitude_fc3_prunclass/cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��Y?�
Eprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Abs/ReadVariableOpReadVariableOpNprune_low_magnitude_fc3_prunclass_cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0�
6prune_low_magnitude_fc3_prunclass/cond/pruning_ops/AbsAbsMprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:y
7prune_low_magnitude_fc3_prunclass/cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value	B : �
7prune_low_magnitude_fc3_prunclass/cond/pruning_ops/CastCast@prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: }
8prune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
6prune_low_magnitude_fc3_prunclass/cond/pruning_ops/subSubAprune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub/x:output:05prune_low_magnitude_fc3_prunclass/cond/Const:output:0*
T0*
_output_shapes
: �
6prune_low_magnitude_fc3_prunclass/cond/pruning_ops/mulMul;prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Cast:y:0:prune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: �
8prune_low_magnitude_fc3_prunclass/cond/pruning_ops/RoundRound:prune_low_magnitude_fc3_prunclass/cond/pruning_ops/mul:z:0*
T0*
_output_shapes
: �
<prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
:prune_low_magnitude_fc3_prunclass/cond/pruning_ops/MaximumMaximum<prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Round:y:0Eprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: �
9prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Cast_1Cast>prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: �
@prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
:prune_low_magnitude_fc3_prunclass/cond/pruning_ops/ReshapeReshape:prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Abs:y:0Iprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes
: {
9prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value	B : �
9prune_low_magnitude_fc3_prunclass/cond/pruning_ops/TopKV2TopKV2Cprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Reshape:output:0Bprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Size_1:output:0*
T0* 
_output_shapes
: : |
:prune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
8prune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub_1Sub=prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Cast_1:y:0Cprune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: �
@prune_low_magnitude_fc3_prunclass/cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;prune_low_magnitude_fc3_prunclass/cond/pruning_ops/GatherV2GatherV2Bprune_low_magnitude_fc3_prunclass/cond/pruning_ops/TopKV2:values:0<prune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub_1:z:0Iprune_low_magnitude_fc3_prunclass/cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: |
:prune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
8prune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub_2Sub=prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Cast_1:y:0Cprune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: �
Bprune_low_magnitude_fc3_prunclass/cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=prune_low_magnitude_fc3_prunclass/cond/pruning_ops/GatherV2_1GatherV2Cprune_low_magnitude_fc3_prunclass/cond/pruning_ops/TopKV2:indices:0<prune_low_magnitude_fc3_prunclass/cond/pruning_ops/sub_2:z:0Kprune_low_magnitude_fc3_prunclass/cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
?prune_low_magnitude_fc3_prunclass/cond/pruning_ops/GreaterEqualGreaterEqual:prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Abs:y:0Dprune_low_magnitude_fc3_prunclass/cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:{
9prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value	B : �
@prune_low_magnitude_fc3_prunclass/cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z�
Bprune_low_magnitude_fc3_prunclass/cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
:prune_low_magnitude_fc3_prunclass/cond/pruning_ops/one_hotOneHotFprune_low_magnitude_fc3_prunclass/cond/pruning_ops/GatherV2_1:output:0Bprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Size_2:output:0Iprune_low_magnitude_fc3_prunclass/cond/pruning_ops/one_hot/Const:output:0Kprune_low_magnitude_fc3_prunclass/cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes
: �
Bprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
<prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Reshape_1ReshapeCprune_low_magnitude_fc3_prunclass/cond/pruning_ops/one_hot:output:0Kprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
<prune_low_magnitude_fc3_prunclass/cond/pruning_ops/LogicalOr	LogicalOrCprune_low_magnitude_fc3_prunclass/cond/pruning_ops/GreaterEqual:z:0Eprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:�
+prune_low_magnitude_fc3_prunclass/cond/CastCast@prune_low_magnitude_fc3_prunclass/cond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
7prune_low_magnitude_fc3_prunclass/cond/AssignVariableOpAssignVariableOp@prune_low_magnitude_fc3_prunclass_cond_assignvariableop_resource/prune_low_magnitude_fc3_prunclass/cond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
9prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp_1AssignVariableOpBprune_low_magnitude_fc3_prunclass_cond_assignvariableop_1_resourceDprune_low_magnitude_fc3_prunclass/cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
1prune_low_magnitude_fc3_prunclass/cond/group_depsNoOp8^prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp:^prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
/prune_low_magnitude_fc3_prunclass/cond/IdentityIdentity^prune_low_magnitude_fc3_prunclass_cond_identity_prune_low_magnitude_fc3_prunclass_logicaland_12^prune_low_magnitude_fc3_prunclass/cond/group_deps*
T0
*
_output_shapes
: �
1prune_low_magnitude_fc3_prunclass/cond/Identity_1Identity8prune_low_magnitude_fc3_prunclass/cond/Identity:output:0,^prune_low_magnitude_fc3_prunclass/cond/NoOp*
T0
*
_output_shapes
: �
+prune_low_magnitude_fc3_prunclass/cond/NoOpNoOp8^prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp:^prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp_1C^prune_low_magnitude_fc3_prunclass/cond/GreaterEqual/ReadVariableOp@^prune_low_magnitude_fc3_prunclass/cond/LessEqual/ReadVariableOp:^prune_low_magnitude_fc3_prunclass/cond/Sub/ReadVariableOpF^prune_low_magnitude_fc3_prunclass/cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "o
1prune_low_magnitude_fc3_prunclass_cond_identity_1:prune_low_magnitude_fc3_prunclass/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2r
7prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp7prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp2v
9prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp_19prune_low_magnitude_fc3_prunclass/cond/AssignVariableOp_12�
Bprune_low_magnitude_fc3_prunclass/cond/GreaterEqual/ReadVariableOpBprune_low_magnitude_fc3_prunclass/cond/GreaterEqual/ReadVariableOp2�
?prune_low_magnitude_fc3_prunclass/cond/LessEqual/ReadVariableOp?prune_low_magnitude_fc3_prunclass/cond/LessEqual/ReadVariableOp2v
9prune_low_magnitude_fc3_prunclass/cond/Sub/ReadVariableOp9prune_low_magnitude_fc3_prunclass/cond/Sub/ReadVariableOp2�
Eprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Abs/ReadVariableOpEprune_low_magnitude_fc3_prunclass/cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�
�
?__inference_prune_low_magnitude_fc2.1_prun_layer_call_fn_495758

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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_493973o
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
�K
�
C__inference_model_1_layer_call_and_return_conditional_losses_494313

inputs

fc1_494232:@

fc1_494234:-
#prune_low_magnitude_fc2_prun_494237:	 5
#prune_low_magnitude_fc2_prun_494239:5
#prune_low_magnitude_fc2_prun_494241:-
#prune_low_magnitude_fc2_prun_494243: 1
#prune_low_magnitude_fc2_prun_494245:/
%prune_low_magnitude_fc2_1_prun_494248:	 7
%prune_low_magnitude_fc2_1_prun_494250:7
%prune_low_magnitude_fc2_1_prun_494252:/
%prune_low_magnitude_fc2_1_prun_494254: 3
%prune_low_magnitude_fc2_1_prun_494256:'
encoder_output_494259:#
encoder_output_494261:/
%prune_low_magnitude_fc3_pruned_494264:	 7
%prune_low_magnitude_fc3_pruned_494266:7
%prune_low_magnitude_fc3_pruned_494268:/
%prune_low_magnitude_fc3_pruned_494270: 3
%prune_low_magnitude_fc3_pruned_494272:2
(prune_low_magnitude_fc3_prunclass_494275:	 :
(prune_low_magnitude_fc3_prunclass_494277::
(prune_low_magnitude_fc3_prunclass_494279:2
(prune_low_magnitude_fc3_prunclass_494281: 6
(prune_low_magnitude_fc3_prunclass_494283:

fc4_494286: 

fc4_494288: "
fc4_class_494291:(
fc4_class_494293:(
fc4_1_494296:  
fc4_1_494298: *
classifier_output_494301:(
&
classifier_output_494303:
&
ecoder_output_494306: @"
ecoder_output_494308:@
identity

identity_1��)classifier_output/StatefulPartitionedCall�%ecoder_output/StatefulPartitionedCall�&encoder_output/StatefulPartitionedCall�fc1/StatefulPartitionedCall�fc4.1/StatefulPartitionedCall�fc4/StatefulPartitionedCall�!fc4_class/StatefulPartitionedCall�6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall�4prune_low_magnitude_fc2_prun/StatefulPartitionedCall�9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall�6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCallinputs
fc1_494232
fc1_494234*
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
GPU 2J 8� *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_493151�
4prune_low_magnitude_fc2_prun/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#prune_low_magnitude_fc2_prun_494237#prune_low_magnitude_fc2_prun_494239#prune_low_magnitude_fc2_prun_494241#prune_low_magnitude_fc2_prun_494243#prune_low_magnitude_fc2_prun_494245*
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
GPU 2J 8� *a
f\RZ
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_494145�
6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_fc2_prun/StatefulPartitionedCall:output:0%prune_low_magnitude_fc2_1_prun_494248%prune_low_magnitude_fc2_1_prun_494250%prune_low_magnitude_fc2_1_prun_494252%prune_low_magnitude_fc2_1_prun_494254%prune_low_magnitude_fc2_1_prun_494256*
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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_493973�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall?prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall:output:0encoder_output_494259encoder_output_494261*
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
GPU 2J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_493214�
6prune_low_magnitude_fc3_pruned/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0%prune_low_magnitude_fc3_pruned_494264%prune_low_magnitude_fc3_pruned_494266%prune_low_magnitude_fc3_pruned_494268%prune_low_magnitude_fc3_pruned_494270%prune_low_magnitude_fc3_pruned_494272*
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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_493791�
9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0(prune_low_magnitude_fc3_prunclass_494275(prune_low_magnitude_fc3_prunclass_494277(prune_low_magnitude_fc3_prunclass_494279(prune_low_magnitude_fc3_prunclass_494281(prune_low_magnitude_fc3_prunclass_494283*
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
GPU 2J 8� *f
faR_
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_493619�
fc4/StatefulPartitionedCallStatefulPartitionedCall?prune_low_magnitude_fc3_pruned/StatefulPartitionedCall:output:0
fc4_494286
fc4_494288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_fc4_layer_call_and_return_conditional_losses_493277�
!fc4_class/StatefulPartitionedCallStatefulPartitionedCallBprune_low_magnitude_fc3_prunclass/StatefulPartitionedCall:output:0fc4_class_494291fc4_class_494293*
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
GPU 2J 8� *N
fIRG
E__inference_fc4_class_layer_call_and_return_conditional_losses_493294�
fc4.1/StatefulPartitionedCallStatefulPartitionedCall$fc4/StatefulPartitionedCall:output:0fc4_1_494296fc4_1_494298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc4.1_layer_call_and_return_conditional_losses_493311�
)classifier_output/StatefulPartitionedCallStatefulPartitionedCall*fc4_class/StatefulPartitionedCall:output:0classifier_output_494301classifier_output_494303*
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
GPU 2J 8� *V
fQRO
M__inference_classifier_output_layer_call_and_return_conditional_losses_493328�
%ecoder_output/StatefulPartitionedCallStatefulPartitionedCall&fc4.1/StatefulPartitionedCall:output:0ecoder_output_494306ecoder_output_494308*
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
GPU 2J 8� *R
fMRK
I__inference_ecoder_output_layer_call_and_return_conditional_losses_493345}
IdentityIdentity.ecoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�

Identity_1Identity2classifier_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp*^classifier_output/StatefulPartitionedCall&^ecoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc4.1/StatefulPartitionedCall^fc4/StatefulPartitionedCall"^fc4_class/StatefulPartitionedCall7^prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall5^prune_low_magnitude_fc2_prun/StatefulPartitionedCall:^prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall7^prune_low_magnitude_fc3_pruned/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)classifier_output/StatefulPartitionedCall)classifier_output/StatefulPartitionedCall2N
%ecoder_output/StatefulPartitionedCall%ecoder_output/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2>
fc4.1/StatefulPartitionedCallfc4.1/StatefulPartitionedCall2:
fc4/StatefulPartitionedCallfc4/StatefulPartitionedCall2F
!fc4_class/StatefulPartitionedCall!fc4_class/StatefulPartitionedCall2p
6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall2l
4prune_low_magnitude_fc2_prun/StatefulPartitionedCall4prune_low_magnitude_fc2_prun/StatefulPartitionedCall2v
9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall2p
6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�K
�
C__inference_model_1_layer_call_and_return_conditional_losses_494613
encoder_input

fc1_494532:@

fc1_494534:-
#prune_low_magnitude_fc2_prun_494537:	 5
#prune_low_magnitude_fc2_prun_494539:5
#prune_low_magnitude_fc2_prun_494541:-
#prune_low_magnitude_fc2_prun_494543: 1
#prune_low_magnitude_fc2_prun_494545:/
%prune_low_magnitude_fc2_1_prun_494548:	 7
%prune_low_magnitude_fc2_1_prun_494550:7
%prune_low_magnitude_fc2_1_prun_494552:/
%prune_low_magnitude_fc2_1_prun_494554: 3
%prune_low_magnitude_fc2_1_prun_494556:'
encoder_output_494559:#
encoder_output_494561:/
%prune_low_magnitude_fc3_pruned_494564:	 7
%prune_low_magnitude_fc3_pruned_494566:7
%prune_low_magnitude_fc3_pruned_494568:/
%prune_low_magnitude_fc3_pruned_494570: 3
%prune_low_magnitude_fc3_pruned_494572:2
(prune_low_magnitude_fc3_prunclass_494575:	 :
(prune_low_magnitude_fc3_prunclass_494577::
(prune_low_magnitude_fc3_prunclass_494579:2
(prune_low_magnitude_fc3_prunclass_494581: 6
(prune_low_magnitude_fc3_prunclass_494583:

fc4_494586: 

fc4_494588: "
fc4_class_494591:(
fc4_class_494593:(
fc4_1_494596:  
fc4_1_494598: *
classifier_output_494601:(
&
classifier_output_494603:
&
ecoder_output_494606: @"
ecoder_output_494608:@
identity

identity_1��)classifier_output/StatefulPartitionedCall�%ecoder_output/StatefulPartitionedCall�&encoder_output/StatefulPartitionedCall�fc1/StatefulPartitionedCall�fc4.1/StatefulPartitionedCall�fc4/StatefulPartitionedCall�!fc4_class/StatefulPartitionedCall�6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall�4prune_low_magnitude_fc2_prun/StatefulPartitionedCall�9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall�6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCallencoder_input
fc1_494532
fc1_494534*
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
GPU 2J 8� *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_493151�
4prune_low_magnitude_fc2_prun/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#prune_low_magnitude_fc2_prun_494537#prune_low_magnitude_fc2_prun_494539#prune_low_magnitude_fc2_prun_494541#prune_low_magnitude_fc2_prun_494543#prune_low_magnitude_fc2_prun_494545*
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
GPU 2J 8� *a
f\RZ
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_494145�
6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_fc2_prun/StatefulPartitionedCall:output:0%prune_low_magnitude_fc2_1_prun_494548%prune_low_magnitude_fc2_1_prun_494550%prune_low_magnitude_fc2_1_prun_494552%prune_low_magnitude_fc2_1_prun_494554%prune_low_magnitude_fc2_1_prun_494556*
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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_493973�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall?prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall:output:0encoder_output_494559encoder_output_494561*
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
GPU 2J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_493214�
6prune_low_magnitude_fc3_pruned/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0%prune_low_magnitude_fc3_pruned_494564%prune_low_magnitude_fc3_pruned_494566%prune_low_magnitude_fc3_pruned_494568%prune_low_magnitude_fc3_pruned_494570%prune_low_magnitude_fc3_pruned_494572*
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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_493791�
9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0(prune_low_magnitude_fc3_prunclass_494575(prune_low_magnitude_fc3_prunclass_494577(prune_low_magnitude_fc3_prunclass_494579(prune_low_magnitude_fc3_prunclass_494581(prune_low_magnitude_fc3_prunclass_494583*
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
GPU 2J 8� *f
faR_
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_493619�
fc4/StatefulPartitionedCallStatefulPartitionedCall?prune_low_magnitude_fc3_pruned/StatefulPartitionedCall:output:0
fc4_494586
fc4_494588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_fc4_layer_call_and_return_conditional_losses_493277�
!fc4_class/StatefulPartitionedCallStatefulPartitionedCallBprune_low_magnitude_fc3_prunclass/StatefulPartitionedCall:output:0fc4_class_494591fc4_class_494593*
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
GPU 2J 8� *N
fIRG
E__inference_fc4_class_layer_call_and_return_conditional_losses_493294�
fc4.1/StatefulPartitionedCallStatefulPartitionedCall$fc4/StatefulPartitionedCall:output:0fc4_1_494596fc4_1_494598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc4.1_layer_call_and_return_conditional_losses_493311�
)classifier_output/StatefulPartitionedCallStatefulPartitionedCall*fc4_class/StatefulPartitionedCall:output:0classifier_output_494601classifier_output_494603*
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
GPU 2J 8� *V
fQRO
M__inference_classifier_output_layer_call_and_return_conditional_losses_493328�
%ecoder_output/StatefulPartitionedCallStatefulPartitionedCall&fc4.1/StatefulPartitionedCall:output:0ecoder_output_494606ecoder_output_494608*
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
GPU 2J 8� *R
fMRK
I__inference_ecoder_output_layer_call_and_return_conditional_losses_493345}
IdentityIdentity.ecoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�

Identity_1Identity2classifier_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp*^classifier_output/StatefulPartitionedCall&^ecoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc4.1/StatefulPartitionedCall^fc4/StatefulPartitionedCall"^fc4_class/StatefulPartitionedCall7^prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall5^prune_low_magnitude_fc2_prun/StatefulPartitionedCall:^prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall7^prune_low_magnitude_fc3_pruned/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)classifier_output/StatefulPartitionedCall)classifier_output/StatefulPartitionedCall2N
%ecoder_output/StatefulPartitionedCall%ecoder_output/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2>
fc4.1/StatefulPartitionedCallfc4.1/StatefulPartitionedCall2:
fc4/StatefulPartitionedCallfc4/StatefulPartitionedCall2F
!fc4_class/StatefulPartitionedCall!fc4_class/StatefulPartitionedCall2p
6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall2l
4prune_low_magnitude_fc2_prun/StatefulPartitionedCall4prune_low_magnitude_fc2_prun/StatefulPartitionedCall2v
9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall2p
6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�F
�
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_493619

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: -
biasadd_readvariableop_resource:
identity��AssignVariableOp�AssignVariableOp_1�BiasAdd/ReadVariableOp�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
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
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_493494*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_493493�
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
 *��Y?�
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
*$
else_branchR
cond_false_493534*
output_shapes
: *#
then_branchR
cond_true_493533q
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
 w
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_120
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
I__inference_ecoder_output_layer_call_and_return_conditional_losses_493345

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
Qprune_low_magnitude_fc2_prun_assert_greater_equal_Assert_AssertGuard_false_494935�
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
�
�
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_495587

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��AssignVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1'
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
 �
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3assert_greater_equal_Assert_AssertGuard_true_495606M
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
ǳ
�
C__inference_model_1_layer_call_and_return_conditional_losses_494908

inputs4
"fc1_matmul_readvariableop_resource:@1
#fc1_biasadd_readvariableop_resource:J
8prune_low_magnitude_fc2_prun_mul_readvariableop_resource:L
:prune_low_magnitude_fc2_prun_mul_readvariableop_1_resource:J
<prune_low_magnitude_fc2_prun_biasadd_readvariableop_resource:L
:prune_low_magnitude_fc2_1_prun_mul_readvariableop_resource:N
<prune_low_magnitude_fc2_1_prun_mul_readvariableop_1_resource:L
>prune_low_magnitude_fc2_1_prun_biasadd_readvariableop_resource:?
-encoder_output_matmul_readvariableop_resource:<
.encoder_output_biasadd_readvariableop_resource:L
:prune_low_magnitude_fc3_pruned_mul_readvariableop_resource:N
<prune_low_magnitude_fc3_pruned_mul_readvariableop_1_resource:L
>prune_low_magnitude_fc3_pruned_biasadd_readvariableop_resource:O
=prune_low_magnitude_fc3_prunclass_mul_readvariableop_resource:Q
?prune_low_magnitude_fc3_prunclass_mul_readvariableop_1_resource:O
Aprune_low_magnitude_fc3_prunclass_biasadd_readvariableop_resource:4
"fc4_matmul_readvariableop_resource: 1
#fc4_biasadd_readvariableop_resource: :
(fc4_class_matmul_readvariableop_resource:(7
)fc4_class_biasadd_readvariableop_resource:(6
$fc4_1_matmul_readvariableop_resource:  3
%fc4_1_biasadd_readvariableop_resource: B
0classifier_output_matmul_readvariableop_resource:(
?
1classifier_output_biasadd_readvariableop_resource:
>
,ecoder_output_matmul_readvariableop_resource: @;
-ecoder_output_biasadd_readvariableop_resource:@
identity

identity_1��(classifier_output/BiasAdd/ReadVariableOp�'classifier_output/MatMul/ReadVariableOp�$ecoder_output/BiasAdd/ReadVariableOp�#ecoder_output/MatMul/ReadVariableOp�%encoder_output/BiasAdd/ReadVariableOp�$encoder_output/MatMul/ReadVariableOp�fc1/BiasAdd/ReadVariableOp�fc1/MatMul/ReadVariableOp�fc4.1/BiasAdd/ReadVariableOp�fc4.1/MatMul/ReadVariableOp�fc4/BiasAdd/ReadVariableOp�fc4/MatMul/ReadVariableOp� fc4_class/BiasAdd/ReadVariableOp�fc4_class/MatMul/ReadVariableOp�/prune_low_magnitude_fc2.1_prun/AssignVariableOp�5prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp�4prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp�1prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp�3prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1�-prune_low_magnitude_fc2_prun/AssignVariableOp�3prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp�2prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp�/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp�1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1�2prune_low_magnitude_fc3_prunclass/AssignVariableOp�8prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp�7prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp�4prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp�6prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1�/prune_low_magnitude_fc3_pruned/AssignVariableOp�5prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp�4prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp�1prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp�3prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1|
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0q

fc1/MatMulMatMulinputs!fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������X
fc1/ReluRelufc1/BiasAdd:output:0*
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
 �
2prune_low_magnitude_fc2_prun/MatMul/ReadVariableOpReadVariableOp8prune_low_magnitude_fc2_prun_mul_readvariableop_resource.^prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
#prune_low_magnitude_fc2_prun/MatMulMatMulfc1/Relu:activations:0:prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
3prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOpReadVariableOp<prune_low_magnitude_fc2_prun_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$prune_low_magnitude_fc2_prun/BiasAddBiasAdd-prune_low_magnitude_fc2_prun/MatMul:product:0;prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!prune_low_magnitude_fc2_prun/ReluRelu-prune_low_magnitude_fc2_prun/BiasAdd:output:0*
T0*'
_output_shapes
:���������F
(prune_low_magnitude_fc2.1_prun/no_updateNoOp*
_output_shapes
 H
*prune_low_magnitude_fc2.1_prun/no_update_1NoOp*
_output_shapes
 �
1prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOpReadVariableOp:prune_low_magnitude_fc2_1_prun_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
3prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1ReadVariableOp<prune_low_magnitude_fc2_1_prun_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc2.1_prun/MulMul9prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp:value:0;prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
/prune_low_magnitude_fc2.1_prun/AssignVariableOpAssignVariableOp:prune_low_magnitude_fc2_1_prun_mul_readvariableop_resource&prune_low_magnitude_fc2.1_prun/Mul:z:02^prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
)prune_low_magnitude_fc2.1_prun/group_depsNoOp0^prune_low_magnitude_fc2.1_prun/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
+prune_low_magnitude_fc2.1_prun/group_deps_1NoOp*^prune_low_magnitude_fc2.1_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
4prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOpReadVariableOp:prune_low_magnitude_fc2_1_prun_mul_readvariableop_resource0^prune_low_magnitude_fc2.1_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
%prune_low_magnitude_fc2.1_prun/MatMulMatMul/prune_low_magnitude_fc2_prun/Relu:activations:0<prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOpReadVariableOp>prune_low_magnitude_fc2_1_prun_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&prune_low_magnitude_fc2.1_prun/BiasAddBiasAdd/prune_low_magnitude_fc2.1_prun/MatMul:product:0=prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#prune_low_magnitude_fc2.1_prun/ReluRelu/prune_low_magnitude_fc2.1_prun/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$encoder_output/MatMul/ReadVariableOpReadVariableOp-encoder_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
encoder_output/MatMulMatMul1prune_low_magnitude_fc2.1_prun/Relu:activations:0,encoder_output/MatMul/ReadVariableOp:value:0*
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
:���������F
(prune_low_magnitude_fc3_pruned/no_updateNoOp*
_output_shapes
 H
*prune_low_magnitude_fc3_pruned/no_update_1NoOp*
_output_shapes
 �
1prune_low_magnitude_fc3_pruned/Mul/ReadVariableOpReadVariableOp:prune_low_magnitude_fc3_pruned_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
3prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1ReadVariableOp<prune_low_magnitude_fc3_pruned_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc3_pruned/MulMul9prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp:value:0;prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
/prune_low_magnitude_fc3_pruned/AssignVariableOpAssignVariableOp:prune_low_magnitude_fc3_pruned_mul_readvariableop_resource&prune_low_magnitude_fc3_pruned/Mul:z:02^prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
)prune_low_magnitude_fc3_pruned/group_depsNoOp0^prune_low_magnitude_fc3_pruned/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
+prune_low_magnitude_fc3_pruned/group_deps_1NoOp*^prune_low_magnitude_fc3_pruned/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
4prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOpReadVariableOp:prune_low_magnitude_fc3_pruned_mul_readvariableop_resource0^prune_low_magnitude_fc3_pruned/AssignVariableOp*
_output_shapes

:*
dtype0�
%prune_low_magnitude_fc3_pruned/MatMulMatMul!encoder_output/Relu:activations:0<prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOpReadVariableOp>prune_low_magnitude_fc3_pruned_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&prune_low_magnitude_fc3_pruned/BiasAddBiasAdd/prune_low_magnitude_fc3_pruned/MatMul:product:0=prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#prune_low_magnitude_fc3_pruned/ReluRelu/prune_low_magnitude_fc3_pruned/BiasAdd:output:0*
T0*'
_output_shapes
:���������I
+prune_low_magnitude_fc3_prunclass/no_updateNoOp*
_output_shapes
 K
-prune_low_magnitude_fc3_prunclass/no_update_1NoOp*
_output_shapes
 �
4prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOpReadVariableOp=prune_low_magnitude_fc3_prunclass_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
6prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1ReadVariableOp?prune_low_magnitude_fc3_prunclass_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
%prune_low_magnitude_fc3_prunclass/MulMul<prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp:value:0>prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
2prune_low_magnitude_fc3_prunclass/AssignVariableOpAssignVariableOp=prune_low_magnitude_fc3_prunclass_mul_readvariableop_resource)prune_low_magnitude_fc3_prunclass/Mul:z:05^prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
,prune_low_magnitude_fc3_prunclass/group_depsNoOp3^prune_low_magnitude_fc3_prunclass/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
.prune_low_magnitude_fc3_prunclass/group_deps_1NoOp-^prune_low_magnitude_fc3_prunclass/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
7prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOpReadVariableOp=prune_low_magnitude_fc3_prunclass_mul_readvariableop_resource3^prune_low_magnitude_fc3_prunclass/AssignVariableOp*
_output_shapes

:*
dtype0�
(prune_low_magnitude_fc3_prunclass/MatMulMatMul!encoder_output/Relu:activations:0?prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
8prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOpReadVariableOpAprune_low_magnitude_fc3_prunclass_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)prune_low_magnitude_fc3_prunclass/BiasAddBiasAdd2prune_low_magnitude_fc3_prunclass/MatMul:product:0@prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&prune_low_magnitude_fc3_prunclass/ReluRelu2prune_low_magnitude_fc3_prunclass/BiasAdd:output:0*
T0*'
_output_shapes
:���������|
fc4/MatMul/ReadVariableOpReadVariableOp"fc4_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�

fc4/MatMulMatMul1prune_low_magnitude_fc3_pruned/Relu:activations:0!fc4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� z
fc4/BiasAdd/ReadVariableOpReadVariableOp#fc4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
fc4/BiasAddBiasAddfc4/MatMul:product:0"fc4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� X
fc4/ReluRelufc4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
fc4_class/MatMul/ReadVariableOpReadVariableOp(fc4_class_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
fc4_class/MatMulMatMul4prune_low_magnitude_fc3_prunclass/Relu:activations:0'fc4_class/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
 fc4_class/BiasAdd/ReadVariableOpReadVariableOp)fc4_class_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
fc4_class/BiasAddBiasAddfc4_class/MatMul:product:0(fc4_class/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(d
fc4_class/ReluRelufc4_class/BiasAdd:output:0*
T0*'
_output_shapes
:���������(�
fc4.1/MatMul/ReadVariableOpReadVariableOp$fc4_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
fc4.1/MatMulMatMulfc4/Relu:activations:0#fc4.1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� ~
fc4.1/BiasAdd/ReadVariableOpReadVariableOp%fc4_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
fc4.1/BiasAddBiasAddfc4.1/MatMul:product:0$fc4.1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� \

fc4.1/ReluRelufc4.1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'classifier_output/MatMul/ReadVariableOpReadVariableOp0classifier_output_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
classifier_output/MatMulMatMulfc4_class/Relu:activations:0/classifier_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
(classifier_output/BiasAdd/ReadVariableOpReadVariableOp1classifier_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
classifier_output/BiasAddBiasAdd"classifier_output/MatMul:product:00classifier_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
z
classifier_output/SoftmaxSoftmax"classifier_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
#ecoder_output/MatMul/ReadVariableOpReadVariableOp,ecoder_output_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
ecoder_output/MatMulMatMulfc4.1/Relu:activations:0+ecoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$ecoder_output/BiasAdd/ReadVariableOpReadVariableOp-ecoder_output_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
ecoder_output/BiasAddBiasAddecoder_output/MatMul:product:0,ecoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
ecoder_output/SigmoidSigmoidecoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������@h
IdentityIdentityecoder_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@t

Identity_1Identity#classifier_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp)^classifier_output/BiasAdd/ReadVariableOp(^classifier_output/MatMul/ReadVariableOp%^ecoder_output/BiasAdd/ReadVariableOp$^ecoder_output/MatMul/ReadVariableOp&^encoder_output/BiasAdd/ReadVariableOp%^encoder_output/MatMul/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc4.1/BiasAdd/ReadVariableOp^fc4.1/MatMul/ReadVariableOp^fc4/BiasAdd/ReadVariableOp^fc4/MatMul/ReadVariableOp!^fc4_class/BiasAdd/ReadVariableOp ^fc4_class/MatMul/ReadVariableOp0^prune_low_magnitude_fc2.1_prun/AssignVariableOp6^prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp5^prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp2^prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp4^prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1.^prune_low_magnitude_fc2_prun/AssignVariableOp4^prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp3^prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp0^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_13^prune_low_magnitude_fc3_prunclass/AssignVariableOp9^prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp8^prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp5^prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp7^prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_10^prune_low_magnitude_fc3_pruned/AssignVariableOp6^prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp5^prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp2^prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp4^prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(classifier_output/BiasAdd/ReadVariableOp(classifier_output/BiasAdd/ReadVariableOp2R
'classifier_output/MatMul/ReadVariableOp'classifier_output/MatMul/ReadVariableOp2L
$ecoder_output/BiasAdd/ReadVariableOp$ecoder_output/BiasAdd/ReadVariableOp2J
#ecoder_output/MatMul/ReadVariableOp#ecoder_output/MatMul/ReadVariableOp2N
%encoder_output/BiasAdd/ReadVariableOp%encoder_output/BiasAdd/ReadVariableOp2L
$encoder_output/MatMul/ReadVariableOp$encoder_output/MatMul/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp2<
fc4.1/BiasAdd/ReadVariableOpfc4.1/BiasAdd/ReadVariableOp2:
fc4.1/MatMul/ReadVariableOpfc4.1/MatMul/ReadVariableOp28
fc4/BiasAdd/ReadVariableOpfc4/BiasAdd/ReadVariableOp26
fc4/MatMul/ReadVariableOpfc4/MatMul/ReadVariableOp2D
 fc4_class/BiasAdd/ReadVariableOp fc4_class/BiasAdd/ReadVariableOp2B
fc4_class/MatMul/ReadVariableOpfc4_class/MatMul/ReadVariableOp2b
/prune_low_magnitude_fc2.1_prun/AssignVariableOp/prune_low_magnitude_fc2.1_prun/AssignVariableOp2n
5prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp5prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp2l
4prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp4prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp2f
1prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp1prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp2j
3prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_13prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_12^
-prune_low_magnitude_fc2_prun/AssignVariableOp-prune_low_magnitude_fc2_prun/AssignVariableOp2j
3prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp3prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp2h
2prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp2prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp2b
/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2f
1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_11prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_12h
2prune_low_magnitude_fc3_prunclass/AssignVariableOp2prune_low_magnitude_fc3_prunclass/AssignVariableOp2t
8prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp8prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp2r
7prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp7prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp2l
4prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp4prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp2p
6prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_16prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_12b
/prune_low_magnitude_fc3_pruned/AssignVariableOp/prune_low_magnitude_fc3_pruned/AssignVariableOp2n
5prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp5prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp2l
4prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp4prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp2f
1prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp1prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp2j
3prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_13prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
?__inference_prune_low_magnitude_fc3_pruned_layer_call_fn_495949

inputs
unknown:
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_493235o
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
�>
�
cond_true_4940593
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
 *��Y?�
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
Uprune_low_magnitude_fc3_prunclass_assert_greater_equal_Assert_AssertGuard_true_495364�
�prune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc3_prunclass_assert_greater_equal_all
Y
Uprune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_placeholder	[
Wprune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_placeholder_1	X
Tprune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_identity_1
�
Nprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Rprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc3_prunclass_assert_greater_equal_allO^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Tprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity_1Identity[prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "�
Tprune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_identity_1]prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
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
�
�
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_493172

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��AssignVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1'
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
 �
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�F
�
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_493973

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: -
biasadd_readvariableop_resource:
identity��AssignVariableOp�AssignVariableOp_1�BiasAdd/ReadVariableOp�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
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
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_493848*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_493847�
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
 *��Y?�
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
*$
else_branchR
cond_false_493888*
output_shapes
: *#
then_branchR
cond_true_493887q
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
 w
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_120
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
J__inference_encoder_output_layer_call_and_return_conditional_losses_493214

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
�
�
4assert_greater_equal_Assert_AssertGuard_false_495793K
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
cond_false_496039
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
�
�
0prune_low_magnitude_fc2.1_prun_cond_false_4951163
/prune_low_magnitude_fc2_1_prun_cond_placeholder5
1prune_low_magnitude_fc2_1_prun_cond_placeholder_15
1prune_low_magnitude_fc2_1_prun_cond_placeholder_25
1prune_low_magnitude_fc2_1_prun_cond_placeholder_3\
Xprune_low_magnitude_fc2_1_prun_cond_identity_prune_low_magnitude_fc2_1_prun_logicaland_1
2
.prune_low_magnitude_fc2_1_prun_cond_identity_1
n
(prune_low_magnitude_fc2.1_prun/cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
,prune_low_magnitude_fc2.1_prun/cond/IdentityIdentityXprune_low_magnitude_fc2_1_prun_cond_identity_prune_low_magnitude_fc2_1_prun_logicaland_1)^prune_low_magnitude_fc2.1_prun/cond/NoOp*
T0
*
_output_shapes
: �
.prune_low_magnitude_fc2.1_prun/cond/Identity_1Identity5prune_low_magnitude_fc2.1_prun/cond/Identity:output:0*
T0
*
_output_shapes
: "i
.prune_low_magnitude_fc2_1_prun_cond_identity_17prune_low_magnitude_fc2.1_prun/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
�F
�
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_493791

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: -
biasadd_readvariableop_resource:
identity��AssignVariableOp�AssignVariableOp_1�BiasAdd/ReadVariableOp�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
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
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_493666*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_493665�
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
 *��Y?�
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
*$
else_branchR
cond_false_493706*
output_shapes
: *#
then_branchR
cond_true_493705q
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
 w
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_120
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�e
�
/prune_low_magnitude_fc2.1_prun_cond_true_495115R
Hprune_low_magnitude_fc2_1_prun_cond_greaterequal_readvariableop_resource:	 ]
Kprune_low_magnitude_fc2_1_prun_cond_pruning_ops_abs_readvariableop_resource:O
=prune_low_magnitude_fc2_1_prun_cond_assignvariableop_resource:I
?prune_low_magnitude_fc2_1_prun_cond_assignvariableop_1_resource: \
Xprune_low_magnitude_fc2_1_prun_cond_identity_prune_low_magnitude_fc2_1_prun_logicaland_1
2
.prune_low_magnitude_fc2_1_prun_cond_identity_1
��4prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp�6prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp_1�?prune_low_magnitude_fc2.1_prun/cond/GreaterEqual/ReadVariableOp�<prune_low_magnitude_fc2.1_prun/cond/LessEqual/ReadVariableOp�6prune_low_magnitude_fc2.1_prun/cond/Sub/ReadVariableOp�Bprune_low_magnitude_fc2.1_prun/cond/pruning_ops/Abs/ReadVariableOp�
?prune_low_magnitude_fc2.1_prun/cond/GreaterEqual/ReadVariableOpReadVariableOpHprune_low_magnitude_fc2_1_prun_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	u
2prune_low_magnitude_fc2.1_prun/cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
0prune_low_magnitude_fc2.1_prun/cond/GreaterEqualGreaterEqualGprune_low_magnitude_fc2.1_prun/cond/GreaterEqual/ReadVariableOp:value:0;prune_low_magnitude_fc2.1_prun/cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
<prune_low_magnitude_fc2.1_prun/cond/LessEqual/ReadVariableOpReadVariableOpHprune_low_magnitude_fc2_1_prun_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	r
/prune_low_magnitude_fc2.1_prun/cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N�
-prune_low_magnitude_fc2.1_prun/cond/LessEqual	LessEqualDprune_low_magnitude_fc2.1_prun/cond/LessEqual/ReadVariableOp:value:08prune_low_magnitude_fc2.1_prun/cond/LessEqual/y:output:0*
T0	*
_output_shapes
: m
*prune_low_magnitude_fc2.1_prun/cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�Nl
*prune_low_magnitude_fc2.1_prun/cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : �
(prune_low_magnitude_fc2.1_prun/cond/LessLess3prune_low_magnitude_fc2.1_prun/cond/Less/x:output:03prune_low_magnitude_fc2.1_prun/cond/Less/y:output:0*
T0*
_output_shapes
: �
-prune_low_magnitude_fc2.1_prun/cond/LogicalOr	LogicalOr1prune_low_magnitude_fc2.1_prun/cond/LessEqual:z:0,prune_low_magnitude_fc2.1_prun/cond/Less:z:0*
_output_shapes
: �
.prune_low_magnitude_fc2.1_prun/cond/LogicalAnd
LogicalAnd4prune_low_magnitude_fc2.1_prun/cond/GreaterEqual:z:01prune_low_magnitude_fc2.1_prun/cond/LogicalOr:z:0*
_output_shapes
: �
6prune_low_magnitude_fc2.1_prun/cond/Sub/ReadVariableOpReadVariableOpHprune_low_magnitude_fc2_1_prun_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	l
)prune_low_magnitude_fc2.1_prun/cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
'prune_low_magnitude_fc2.1_prun/cond/SubSub>prune_low_magnitude_fc2.1_prun/cond/Sub/ReadVariableOp:value:02prune_low_magnitude_fc2.1_prun/cond/Sub/y:output:0*
T0	*
_output_shapes
: p
.prune_low_magnitude_fc2.1_prun/cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd�
,prune_low_magnitude_fc2.1_prun/cond/FloorModFloorMod+prune_low_magnitude_fc2.1_prun/cond/Sub:z:07prune_low_magnitude_fc2.1_prun/cond/FloorMod/y:output:0*
T0	*
_output_shapes
: m
+prune_low_magnitude_fc2.1_prun/cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
)prune_low_magnitude_fc2.1_prun/cond/EqualEqual0prune_low_magnitude_fc2.1_prun/cond/FloorMod:z:04prune_low_magnitude_fc2.1_prun/cond/Equal/y:output:0*
T0	*
_output_shapes
: �
0prune_low_magnitude_fc2.1_prun/cond/LogicalAnd_1
LogicalAnd2prune_low_magnitude_fc2.1_prun/cond/LogicalAnd:z:0-prune_low_magnitude_fc2.1_prun/cond/Equal:z:0*
_output_shapes
: n
)prune_low_magnitude_fc2.1_prun/cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��Y?�
Bprune_low_magnitude_fc2.1_prun/cond/pruning_ops/Abs/ReadVariableOpReadVariableOpKprune_low_magnitude_fc2_1_prun_cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0�
3prune_low_magnitude_fc2.1_prun/cond/pruning_ops/AbsAbsJprune_low_magnitude_fc2.1_prun/cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:w
4prune_low_magnitude_fc2.1_prun/cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :��
4prune_low_magnitude_fc2.1_prun/cond/pruning_ops/CastCast=prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: z
5prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3prune_low_magnitude_fc2.1_prun/cond/pruning_ops/subSub>prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub/x:output:02prune_low_magnitude_fc2.1_prun/cond/Const:output:0*
T0*
_output_shapes
: �
3prune_low_magnitude_fc2.1_prun/cond/pruning_ops/mulMul8prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Cast:y:07prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: �
5prune_low_magnitude_fc2.1_prun/cond/pruning_ops/RoundRound7prune_low_magnitude_fc2.1_prun/cond/pruning_ops/mul:z:0*
T0*
_output_shapes
: ~
9prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7prune_low_magnitude_fc2.1_prun/cond/pruning_ops/MaximumMaximum9prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Round:y:0Bprune_low_magnitude_fc2.1_prun/cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: �
6prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Cast_1Cast;prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: �
=prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7prune_low_magnitude_fc2.1_prun/cond/pruning_ops/ReshapeReshape7prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Abs:y:0Fprune_low_magnitude_fc2.1_prun/cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:�y
6prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :��
6prune_low_magnitude_fc2.1_prun/cond/pruning_ops/TopKV2TopKV2@prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Reshape:output:0?prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:�:�y
7prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
5prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub_1Sub:prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Cast_1:y:0@prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 
=prune_low_magnitude_fc2.1_prun/cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8prune_low_magnitude_fc2.1_prun/cond/pruning_ops/GatherV2GatherV2?prune_low_magnitude_fc2.1_prun/cond/pruning_ops/TopKV2:values:09prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub_1:z:0Fprune_low_magnitude_fc2.1_prun/cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: y
7prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
5prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub_2Sub:prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Cast_1:y:0@prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: �
?prune_low_magnitude_fc2.1_prun/cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:prune_low_magnitude_fc2.1_prun/cond/pruning_ops/GatherV2_1GatherV2@prune_low_magnitude_fc2.1_prun/cond/pruning_ops/TopKV2:indices:09prune_low_magnitude_fc2.1_prun/cond/pruning_ops/sub_2:z:0Hprune_low_magnitude_fc2.1_prun/cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
<prune_low_magnitude_fc2.1_prun/cond/pruning_ops/GreaterEqualGreaterEqual7prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Abs:y:0Aprune_low_magnitude_fc2.1_prun/cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:y
6prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value
B :�
=prune_low_magnitude_fc2.1_prun/cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z�
?prune_low_magnitude_fc2.1_prun/cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
7prune_low_magnitude_fc2.1_prun/cond/pruning_ops/one_hotOneHotCprune_low_magnitude_fc2.1_prun/cond/pruning_ops/GatherV2_1:output:0?prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Size_2:output:0Fprune_low_magnitude_fc2.1_prun/cond/pruning_ops/one_hot/Const:output:0Hprune_low_magnitude_fc2.1_prun/cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes	
:��
?prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
9prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Reshape_1Reshape@prune_low_magnitude_fc2.1_prun/cond/pruning_ops/one_hot:output:0Hprune_low_magnitude_fc2.1_prun/cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
9prune_low_magnitude_fc2.1_prun/cond/pruning_ops/LogicalOr	LogicalOr@prune_low_magnitude_fc2.1_prun/cond/pruning_ops/GreaterEqual:z:0Bprune_low_magnitude_fc2.1_prun/cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:�
(prune_low_magnitude_fc2.1_prun/cond/CastCast=prune_low_magnitude_fc2.1_prun/cond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
4prune_low_magnitude_fc2.1_prun/cond/AssignVariableOpAssignVariableOp=prune_low_magnitude_fc2_1_prun_cond_assignvariableop_resource,prune_low_magnitude_fc2.1_prun/cond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
6prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp_1AssignVariableOp?prune_low_magnitude_fc2_1_prun_cond_assignvariableop_1_resourceAprune_low_magnitude_fc2.1_prun/cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
.prune_low_magnitude_fc2.1_prun/cond/group_depsNoOp5^prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp7^prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
,prune_low_magnitude_fc2.1_prun/cond/IdentityIdentityXprune_low_magnitude_fc2_1_prun_cond_identity_prune_low_magnitude_fc2_1_prun_logicaland_1/^prune_low_magnitude_fc2.1_prun/cond/group_deps*
T0
*
_output_shapes
: �
.prune_low_magnitude_fc2.1_prun/cond/Identity_1Identity5prune_low_magnitude_fc2.1_prun/cond/Identity:output:0)^prune_low_magnitude_fc2.1_prun/cond/NoOp*
T0
*
_output_shapes
: �
(prune_low_magnitude_fc2.1_prun/cond/NoOpNoOp5^prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp7^prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp_1@^prune_low_magnitude_fc2.1_prun/cond/GreaterEqual/ReadVariableOp=^prune_low_magnitude_fc2.1_prun/cond/LessEqual/ReadVariableOp7^prune_low_magnitude_fc2.1_prun/cond/Sub/ReadVariableOpC^prune_low_magnitude_fc2.1_prun/cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "i
.prune_low_magnitude_fc2_1_prun_cond_identity_17prune_low_magnitude_fc2.1_prun/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2l
4prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp4prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp2p
6prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp_16prune_low_magnitude_fc2.1_prun/cond/AssignVariableOp_12�
?prune_low_magnitude_fc2.1_prun/cond/GreaterEqual/ReadVariableOp?prune_low_magnitude_fc2.1_prun/cond/GreaterEqual/ReadVariableOp2|
<prune_low_magnitude_fc2.1_prun/cond/LessEqual/ReadVariableOp<prune_low_magnitude_fc2.1_prun/cond/LessEqual/ReadVariableOp2p
6prune_low_magnitude_fc2.1_prun/cond/Sub/ReadVariableOp6prune_low_magnitude_fc2.1_prun/cond/Sub/ReadVariableOp2�
Bprune_low_magnitude_fc2.1_prun/cond/pruning_ops/Abs/ReadVariableOpBprune_low_magnitude_fc2.1_prun/cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�

�
M__inference_classifier_output_layer_call_and_return_conditional_losses_493328

inputs0
matmul_readvariableop_resource:(
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
3assert_greater_equal_Assert_AssertGuard_true_495998M
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
cond_false_493706
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
�
�
cond_false_495833
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
cond_true_4935333
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
 *��Y?�
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
�
�
(__inference_model_1_layer_call_fn_494461
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

unknown_17:

unknown_18:	 

unknown_19:

unknown_20:

unknown_21: 

unknown_22:

unknown_23: 

unknown_24: 

unknown_25:(

unknown_26:(

unknown_27:  

unknown_28: 

unknown_29:(


unknown_30:


unknown_31: @

unknown_32:@
identity

identity_1��StatefulPartitionedCall�
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
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������
*4
_read_only_resource_inputs
 !"*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_494313o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�
�
4assert_greater_equal_Assert_AssertGuard_false_493494K
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
�
�
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_493235

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��AssignVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1'
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
 �
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�F
�
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_496124

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: -
biasadd_readvariableop_resource:
identity��AssignVariableOp�AssignVariableOp_1�BiasAdd/ReadVariableOp�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
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
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_495999*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_495998�
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
 *��Y?�
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
*$
else_branchR
cond_false_496039*
output_shapes
: *#
then_branchR
cond_true_496038q
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
 w
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_120
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4assert_greater_equal_Assert_AssertGuard_false_493666K
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
�
�
B__inference_prune_low_magnitude_fc3_prunclass_layer_call_fn_496155

inputs
unknown:
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
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
GPU 2J 8� *f
faR_
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_493258o
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
�
�
/__inference_encoder_output_layer_call_fn_495927

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
GPU 2J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_493214o
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
�
�
=__inference_prune_low_magnitude_fc2_prun_layer_call_fn_495557

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
GPU 2J 8� *a
f\RZ
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_493172o
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
�e
�
/prune_low_magnitude_fc3_pruned_cond_true_495263R
Hprune_low_magnitude_fc3_pruned_cond_greaterequal_readvariableop_resource:	 ]
Kprune_low_magnitude_fc3_pruned_cond_pruning_ops_abs_readvariableop_resource:O
=prune_low_magnitude_fc3_pruned_cond_assignvariableop_resource:I
?prune_low_magnitude_fc3_pruned_cond_assignvariableop_1_resource: \
Xprune_low_magnitude_fc3_pruned_cond_identity_prune_low_magnitude_fc3_pruned_logicaland_1
2
.prune_low_magnitude_fc3_pruned_cond_identity_1
��4prune_low_magnitude_fc3_pruned/cond/AssignVariableOp�6prune_low_magnitude_fc3_pruned/cond/AssignVariableOp_1�?prune_low_magnitude_fc3_pruned/cond/GreaterEqual/ReadVariableOp�<prune_low_magnitude_fc3_pruned/cond/LessEqual/ReadVariableOp�6prune_low_magnitude_fc3_pruned/cond/Sub/ReadVariableOp�Bprune_low_magnitude_fc3_pruned/cond/pruning_ops/Abs/ReadVariableOp�
?prune_low_magnitude_fc3_pruned/cond/GreaterEqual/ReadVariableOpReadVariableOpHprune_low_magnitude_fc3_pruned_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	u
2prune_low_magnitude_fc3_pruned/cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
0prune_low_magnitude_fc3_pruned/cond/GreaterEqualGreaterEqualGprune_low_magnitude_fc3_pruned/cond/GreaterEqual/ReadVariableOp:value:0;prune_low_magnitude_fc3_pruned/cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
<prune_low_magnitude_fc3_pruned/cond/LessEqual/ReadVariableOpReadVariableOpHprune_low_magnitude_fc3_pruned_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	r
/prune_low_magnitude_fc3_pruned/cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
value
B	 R�N�
-prune_low_magnitude_fc3_pruned/cond/LessEqual	LessEqualDprune_low_magnitude_fc3_pruned/cond/LessEqual/ReadVariableOp:value:08prune_low_magnitude_fc3_pruned/cond/LessEqual/y:output:0*
T0	*
_output_shapes
: m
*prune_low_magnitude_fc3_pruned/cond/Less/xConst*
_output_shapes
: *
dtype0*
value
B :�Nl
*prune_low_magnitude_fc3_pruned/cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : �
(prune_low_magnitude_fc3_pruned/cond/LessLess3prune_low_magnitude_fc3_pruned/cond/Less/x:output:03prune_low_magnitude_fc3_pruned/cond/Less/y:output:0*
T0*
_output_shapes
: �
-prune_low_magnitude_fc3_pruned/cond/LogicalOr	LogicalOr1prune_low_magnitude_fc3_pruned/cond/LessEqual:z:0,prune_low_magnitude_fc3_pruned/cond/Less:z:0*
_output_shapes
: �
.prune_low_magnitude_fc3_pruned/cond/LogicalAnd
LogicalAnd4prune_low_magnitude_fc3_pruned/cond/GreaterEqual:z:01prune_low_magnitude_fc3_pruned/cond/LogicalOr:z:0*
_output_shapes
: �
6prune_low_magnitude_fc3_pruned/cond/Sub/ReadVariableOpReadVariableOpHprune_low_magnitude_fc3_pruned_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	l
)prune_low_magnitude_fc3_pruned/cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value
B	 R��
'prune_low_magnitude_fc3_pruned/cond/SubSub>prune_low_magnitude_fc3_pruned/cond/Sub/ReadVariableOp:value:02prune_low_magnitude_fc3_pruned/cond/Sub/y:output:0*
T0	*
_output_shapes
: p
.prune_low_magnitude_fc3_pruned/cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd�
,prune_low_magnitude_fc3_pruned/cond/FloorModFloorMod+prune_low_magnitude_fc3_pruned/cond/Sub:z:07prune_low_magnitude_fc3_pruned/cond/FloorMod/y:output:0*
T0	*
_output_shapes
: m
+prune_low_magnitude_fc3_pruned/cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
)prune_low_magnitude_fc3_pruned/cond/EqualEqual0prune_low_magnitude_fc3_pruned/cond/FloorMod:z:04prune_low_magnitude_fc3_pruned/cond/Equal/y:output:0*
T0	*
_output_shapes
: �
0prune_low_magnitude_fc3_pruned/cond/LogicalAnd_1
LogicalAnd2prune_low_magnitude_fc3_pruned/cond/LogicalAnd:z:0-prune_low_magnitude_fc3_pruned/cond/Equal:z:0*
_output_shapes
: n
)prune_low_magnitude_fc3_pruned/cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��Y?�
Bprune_low_magnitude_fc3_pruned/cond/pruning_ops/Abs/ReadVariableOpReadVariableOpKprune_low_magnitude_fc3_pruned_cond_pruning_ops_abs_readvariableop_resource*
_output_shapes

:*
dtype0�
3prune_low_magnitude_fc3_pruned/cond/pruning_ops/AbsAbsJprune_low_magnitude_fc3_pruned/cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:v
4prune_low_magnitude_fc3_pruned/cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value	B : �
4prune_low_magnitude_fc3_pruned/cond/pruning_ops/CastCast=prune_low_magnitude_fc3_pruned/cond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: z
5prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3prune_low_magnitude_fc3_pruned/cond/pruning_ops/subSub>prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub/x:output:02prune_low_magnitude_fc3_pruned/cond/Const:output:0*
T0*
_output_shapes
: �
3prune_low_magnitude_fc3_pruned/cond/pruning_ops/mulMul8prune_low_magnitude_fc3_pruned/cond/pruning_ops/Cast:y:07prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: �
5prune_low_magnitude_fc3_pruned/cond/pruning_ops/RoundRound7prune_low_magnitude_fc3_pruned/cond/pruning_ops/mul:z:0*
T0*
_output_shapes
: ~
9prune_low_magnitude_fc3_pruned/cond/pruning_ops/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7prune_low_magnitude_fc3_pruned/cond/pruning_ops/MaximumMaximum9prune_low_magnitude_fc3_pruned/cond/pruning_ops/Round:y:0Bprune_low_magnitude_fc3_pruned/cond/pruning_ops/Maximum/y:output:0*
T0*
_output_shapes
: �
6prune_low_magnitude_fc3_pruned/cond/pruning_ops/Cast_1Cast;prune_low_magnitude_fc3_pruned/cond/pruning_ops/Maximum:z:0*

DstT0*

SrcT0*
_output_shapes
: �
=prune_low_magnitude_fc3_pruned/cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7prune_low_magnitude_fc3_pruned/cond/pruning_ops/ReshapeReshape7prune_low_magnitude_fc3_pruned/cond/pruning_ops/Abs:y:0Fprune_low_magnitude_fc3_pruned/cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes
: x
6prune_low_magnitude_fc3_pruned/cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value	B : �
6prune_low_magnitude_fc3_pruned/cond/pruning_ops/TopKV2TopKV2@prune_low_magnitude_fc3_pruned/cond/pruning_ops/Reshape:output:0?prune_low_magnitude_fc3_pruned/cond/pruning_ops/Size_1:output:0*
T0* 
_output_shapes
: : y
7prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
5prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub_1Sub:prune_low_magnitude_fc3_pruned/cond/pruning_ops/Cast_1:y:0@prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 
=prune_low_magnitude_fc3_pruned/cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8prune_low_magnitude_fc3_pruned/cond/pruning_ops/GatherV2GatherV2?prune_low_magnitude_fc3_pruned/cond/pruning_ops/TopKV2:values:09prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub_1:z:0Fprune_low_magnitude_fc3_pruned/cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: y
7prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
5prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub_2Sub:prune_low_magnitude_fc3_pruned/cond/pruning_ops/Cast_1:y:0@prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub_2/y:output:0*
T0*
_output_shapes
: �
?prune_low_magnitude_fc3_pruned/cond/pruning_ops/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:prune_low_magnitude_fc3_pruned/cond/pruning_ops/GatherV2_1GatherV2@prune_low_magnitude_fc3_pruned/cond/pruning_ops/TopKV2:indices:09prune_low_magnitude_fc3_pruned/cond/pruning_ops/sub_2:z:0Hprune_low_magnitude_fc3_pruned/cond/pruning_ops/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: �
<prune_low_magnitude_fc3_pruned/cond/pruning_ops/GreaterEqualGreaterEqual7prune_low_magnitude_fc3_pruned/cond/pruning_ops/Abs:y:0Aprune_low_magnitude_fc3_pruned/cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes

:x
6prune_low_magnitude_fc3_pruned/cond/pruning_ops/Size_2Const*
_output_shapes
: *
dtype0*
value	B : 
=prune_low_magnitude_fc3_pruned/cond/pruning_ops/one_hot/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z�
?prune_low_magnitude_fc3_pruned/cond/pruning_ops/one_hot/Const_1Const*
_output_shapes
: *
dtype0
*
value	B
 Z �
7prune_low_magnitude_fc3_pruned/cond/pruning_ops/one_hotOneHotCprune_low_magnitude_fc3_pruned/cond/pruning_ops/GatherV2_1:output:0?prune_low_magnitude_fc3_pruned/cond/pruning_ops/Size_2:output:0Fprune_low_magnitude_fc3_pruned/cond/pruning_ops/one_hot/Const:output:0Hprune_low_magnitude_fc3_pruned/cond/pruning_ops/one_hot/Const_1:output:0*
T0
*
TI0*
_output_shapes
: �
?prune_low_magnitude_fc3_pruned/cond/pruning_ops/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
9prune_low_magnitude_fc3_pruned/cond/pruning_ops/Reshape_1Reshape@prune_low_magnitude_fc3_pruned/cond/pruning_ops/one_hot:output:0Hprune_low_magnitude_fc3_pruned/cond/pruning_ops/Reshape_1/shape:output:0*
T0
*
_output_shapes

:�
9prune_low_magnitude_fc3_pruned/cond/pruning_ops/LogicalOr	LogicalOr@prune_low_magnitude_fc3_pruned/cond/pruning_ops/GreaterEqual:z:0Bprune_low_magnitude_fc3_pruned/cond/pruning_ops/Reshape_1:output:0*
_output_shapes

:�
(prune_low_magnitude_fc3_pruned/cond/CastCast=prune_low_magnitude_fc3_pruned/cond/pruning_ops/LogicalOr:z:0*

DstT0*

SrcT0
*
_output_shapes

:�
4prune_low_magnitude_fc3_pruned/cond/AssignVariableOpAssignVariableOp=prune_low_magnitude_fc3_pruned_cond_assignvariableop_resource,prune_low_magnitude_fc3_pruned/cond/Cast:y:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
6prune_low_magnitude_fc3_pruned/cond/AssignVariableOp_1AssignVariableOp?prune_low_magnitude_fc3_pruned_cond_assignvariableop_1_resourceAprune_low_magnitude_fc3_pruned/cond/pruning_ops/GatherV2:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
.prune_low_magnitude_fc3_pruned/cond/group_depsNoOp5^prune_low_magnitude_fc3_pruned/cond/AssignVariableOp7^prune_low_magnitude_fc3_pruned/cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
,prune_low_magnitude_fc3_pruned/cond/IdentityIdentityXprune_low_magnitude_fc3_pruned_cond_identity_prune_low_magnitude_fc3_pruned_logicaland_1/^prune_low_magnitude_fc3_pruned/cond/group_deps*
T0
*
_output_shapes
: �
.prune_low_magnitude_fc3_pruned/cond/Identity_1Identity5prune_low_magnitude_fc3_pruned/cond/Identity:output:0)^prune_low_magnitude_fc3_pruned/cond/NoOp*
T0
*
_output_shapes
: �
(prune_low_magnitude_fc3_pruned/cond/NoOpNoOp5^prune_low_magnitude_fc3_pruned/cond/AssignVariableOp7^prune_low_magnitude_fc3_pruned/cond/AssignVariableOp_1@^prune_low_magnitude_fc3_pruned/cond/GreaterEqual/ReadVariableOp=^prune_low_magnitude_fc3_pruned/cond/LessEqual/ReadVariableOp7^prune_low_magnitude_fc3_pruned/cond/Sub/ReadVariableOpC^prune_low_magnitude_fc3_pruned/cond/pruning_ops/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "i
.prune_low_magnitude_fc3_pruned_cond_identity_17prune_low_magnitude_fc3_pruned/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2l
4prune_low_magnitude_fc3_pruned/cond/AssignVariableOp4prune_low_magnitude_fc3_pruned/cond/AssignVariableOp2p
6prune_low_magnitude_fc3_pruned/cond/AssignVariableOp_16prune_low_magnitude_fc3_pruned/cond/AssignVariableOp_12�
?prune_low_magnitude_fc3_pruned/cond/GreaterEqual/ReadVariableOp?prune_low_magnitude_fc3_pruned/cond/GreaterEqual/ReadVariableOp2|
<prune_low_magnitude_fc3_pruned/cond/LessEqual/ReadVariableOp<prune_low_magnitude_fc3_pruned/cond/LessEqual/ReadVariableOp2p
6prune_low_magnitude_fc3_pruned/cond/Sub/ReadVariableOp6prune_low_magnitude_fc3_pruned/cond/Sub/ReadVariableOp2�
Bprune_low_magnitude_fc3_pruned/cond/pruning_ops/Abs/ReadVariableOpBprune_low_magnitude_fc3_pruned/cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
�
�
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_495773

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��AssignVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1'
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
 �
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
I__inference_ecoder_output_layer_call_and_return_conditional_losses_496390

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
.__inference_ecoder_output_layer_call_fn_496379

inputs
unknown: @
	unknown_0:@
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
GPU 2J 8� *R
fMRK
I__inference_ecoder_output_layer_call_and_return_conditional_losses_493345o
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
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
(__inference_model_1_layer_call_fn_494810

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

unknown_17:

unknown_18:	 

unknown_19:

unknown_20:

unknown_21: 

unknown_22:

unknown_23: 

unknown_24: 

unknown_25:(

unknown_26:(

unknown_27:  

unknown_28: 

unknown_29:(


unknown_30:


unknown_31: @

unknown_32:@
identity

identity_1��StatefulPartitionedCall�
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
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������
*4
_read_only_resource_inputs
 !"*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_494313o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�>
�
cond_true_4960383
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
 *��Y?�
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
��
�
!__inference__wrapped_model_493133
encoder_input<
*model_1_fc1_matmul_readvariableop_resource:@9
+model_1_fc1_biasadd_readvariableop_resource:R
@model_1_prune_low_magnitude_fc2_prun_mul_readvariableop_resource:T
Bmodel_1_prune_low_magnitude_fc2_prun_mul_readvariableop_1_resource:R
Dmodel_1_prune_low_magnitude_fc2_prun_biasadd_readvariableop_resource:T
Bmodel_1_prune_low_magnitude_fc2_1_prun_mul_readvariableop_resource:V
Dmodel_1_prune_low_magnitude_fc2_1_prun_mul_readvariableop_1_resource:T
Fmodel_1_prune_low_magnitude_fc2_1_prun_biasadd_readvariableop_resource:G
5model_1_encoder_output_matmul_readvariableop_resource:D
6model_1_encoder_output_biasadd_readvariableop_resource:T
Bmodel_1_prune_low_magnitude_fc3_pruned_mul_readvariableop_resource:V
Dmodel_1_prune_low_magnitude_fc3_pruned_mul_readvariableop_1_resource:T
Fmodel_1_prune_low_magnitude_fc3_pruned_biasadd_readvariableop_resource:W
Emodel_1_prune_low_magnitude_fc3_prunclass_mul_readvariableop_resource:Y
Gmodel_1_prune_low_magnitude_fc3_prunclass_mul_readvariableop_1_resource:W
Imodel_1_prune_low_magnitude_fc3_prunclass_biasadd_readvariableop_resource:<
*model_1_fc4_matmul_readvariableop_resource: 9
+model_1_fc4_biasadd_readvariableop_resource: B
0model_1_fc4_class_matmul_readvariableop_resource:(?
1model_1_fc4_class_biasadd_readvariableop_resource:(>
,model_1_fc4_1_matmul_readvariableop_resource:  ;
-model_1_fc4_1_biasadd_readvariableop_resource: J
8model_1_classifier_output_matmul_readvariableop_resource:(
G
9model_1_classifier_output_biasadd_readvariableop_resource:
F
4model_1_ecoder_output_matmul_readvariableop_resource: @C
5model_1_ecoder_output_biasadd_readvariableop_resource:@
identity

identity_1��0model_1/classifier_output/BiasAdd/ReadVariableOp�/model_1/classifier_output/MatMul/ReadVariableOp�,model_1/ecoder_output/BiasAdd/ReadVariableOp�+model_1/ecoder_output/MatMul/ReadVariableOp�-model_1/encoder_output/BiasAdd/ReadVariableOp�,model_1/encoder_output/MatMul/ReadVariableOp�"model_1/fc1/BiasAdd/ReadVariableOp�!model_1/fc1/MatMul/ReadVariableOp�$model_1/fc4.1/BiasAdd/ReadVariableOp�#model_1/fc4.1/MatMul/ReadVariableOp�"model_1/fc4/BiasAdd/ReadVariableOp�!model_1/fc4/MatMul/ReadVariableOp�(model_1/fc4_class/BiasAdd/ReadVariableOp�'model_1/fc4_class/MatMul/ReadVariableOp�7model_1/prune_low_magnitude_fc2.1_prun/AssignVariableOp�=model_1/prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp�<model_1/prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp�9model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp�;model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1�5model_1/prune_low_magnitude_fc2_prun/AssignVariableOp�;model_1/prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp�:model_1/prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp�7model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp�9model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1�:model_1/prune_low_magnitude_fc3_prunclass/AssignVariableOp�@model_1/prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp�?model_1/prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp�<model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp�>model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1�7model_1/prune_low_magnitude_fc3_pruned/AssignVariableOp�=model_1/prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp�<model_1/prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp�9model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp�;model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1�
!model_1/fc1/MatMul/ReadVariableOpReadVariableOp*model_1_fc1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_1/fc1/MatMulMatMulencoder_input)model_1/fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model_1/fc1/BiasAdd/ReadVariableOpReadVariableOp+model_1_fc1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/fc1/BiasAddBiasAddmodel_1/fc1/MatMul:product:0*model_1/fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
model_1/fc1/ReluRelumodel_1/fc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������L
.model_1/prune_low_magnitude_fc2_prun/no_updateNoOp*
_output_shapes
 N
0model_1/prune_low_magnitude_fc2_prun/no_update_1NoOp*
_output_shapes
 �
7model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOpReadVariableOp@model_1_prune_low_magnitude_fc2_prun_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
9model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1ReadVariableOpBmodel_1_prune_low_magnitude_fc2_prun_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
(model_1/prune_low_magnitude_fc2_prun/MulMul?model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp:value:0Amodel_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
5model_1/prune_low_magnitude_fc2_prun/AssignVariableOpAssignVariableOp@model_1_prune_low_magnitude_fc2_prun_mul_readvariableop_resource,model_1/prune_low_magnitude_fc2_prun/Mul:z:08^model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
/model_1/prune_low_magnitude_fc2_prun/group_depsNoOp6^model_1/prune_low_magnitude_fc2_prun/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
1model_1/prune_low_magnitude_fc2_prun/group_deps_1NoOp0^model_1/prune_low_magnitude_fc2_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
:model_1/prune_low_magnitude_fc2_prun/MatMul/ReadVariableOpReadVariableOp@model_1_prune_low_magnitude_fc2_prun_mul_readvariableop_resource6^model_1/prune_low_magnitude_fc2_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
+model_1/prune_low_magnitude_fc2_prun/MatMulMatMulmodel_1/fc1/Relu:activations:0Bmodel_1/prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
;model_1/prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOpReadVariableOpDmodel_1_prune_low_magnitude_fc2_prun_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_1/prune_low_magnitude_fc2_prun/BiasAddBiasAdd5model_1/prune_low_magnitude_fc2_prun/MatMul:product:0Cmodel_1/prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)model_1/prune_low_magnitude_fc2_prun/ReluRelu5model_1/prune_low_magnitude_fc2_prun/BiasAdd:output:0*
T0*'
_output_shapes
:���������N
0model_1/prune_low_magnitude_fc2.1_prun/no_updateNoOp*
_output_shapes
 P
2model_1/prune_low_magnitude_fc2.1_prun/no_update_1NoOp*
_output_shapes
 �
9model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOpReadVariableOpBmodel_1_prune_low_magnitude_fc2_1_prun_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
;model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1ReadVariableOpDmodel_1_prune_low_magnitude_fc2_1_prun_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
*model_1/prune_low_magnitude_fc2.1_prun/MulMulAmodel_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp:value:0Cmodel_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
7model_1/prune_low_magnitude_fc2.1_prun/AssignVariableOpAssignVariableOpBmodel_1_prune_low_magnitude_fc2_1_prun_mul_readvariableop_resource.model_1/prune_low_magnitude_fc2.1_prun/Mul:z:0:^model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
1model_1/prune_low_magnitude_fc2.1_prun/group_depsNoOp8^model_1/prune_low_magnitude_fc2.1_prun/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
3model_1/prune_low_magnitude_fc2.1_prun/group_deps_1NoOp2^model_1/prune_low_magnitude_fc2.1_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
<model_1/prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOpReadVariableOpBmodel_1_prune_low_magnitude_fc2_1_prun_mul_readvariableop_resource8^model_1/prune_low_magnitude_fc2.1_prun/AssignVariableOp*
_output_shapes

:*
dtype0�
-model_1/prune_low_magnitude_fc2.1_prun/MatMulMatMul7model_1/prune_low_magnitude_fc2_prun/Relu:activations:0Dmodel_1/prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
=model_1/prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_prune_low_magnitude_fc2_1_prun_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
.model_1/prune_low_magnitude_fc2.1_prun/BiasAddBiasAdd7model_1/prune_low_magnitude_fc2.1_prun/MatMul:product:0Emodel_1/prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+model_1/prune_low_magnitude_fc2.1_prun/ReluRelu7model_1/prune_low_magnitude_fc2.1_prun/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,model_1/encoder_output/MatMul/ReadVariableOpReadVariableOp5model_1_encoder_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_1/encoder_output/MatMulMatMul9model_1/prune_low_magnitude_fc2.1_prun/Relu:activations:04model_1/encoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-model_1/encoder_output/BiasAdd/ReadVariableOpReadVariableOp6model_1_encoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/encoder_output/BiasAddBiasAdd'model_1/encoder_output/MatMul:product:05model_1/encoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
model_1/encoder_output/ReluRelu'model_1/encoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������N
0model_1/prune_low_magnitude_fc3_pruned/no_updateNoOp*
_output_shapes
 P
2model_1/prune_low_magnitude_fc3_pruned/no_update_1NoOp*
_output_shapes
 �
9model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOpReadVariableOpBmodel_1_prune_low_magnitude_fc3_pruned_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
;model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1ReadVariableOpDmodel_1_prune_low_magnitude_fc3_pruned_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
*model_1/prune_low_magnitude_fc3_pruned/MulMulAmodel_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp:value:0Cmodel_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
7model_1/prune_low_magnitude_fc3_pruned/AssignVariableOpAssignVariableOpBmodel_1_prune_low_magnitude_fc3_pruned_mul_readvariableop_resource.model_1/prune_low_magnitude_fc3_pruned/Mul:z:0:^model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
1model_1/prune_low_magnitude_fc3_pruned/group_depsNoOp8^model_1/prune_low_magnitude_fc3_pruned/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
3model_1/prune_low_magnitude_fc3_pruned/group_deps_1NoOp2^model_1/prune_low_magnitude_fc3_pruned/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
<model_1/prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOpReadVariableOpBmodel_1_prune_low_magnitude_fc3_pruned_mul_readvariableop_resource8^model_1/prune_low_magnitude_fc3_pruned/AssignVariableOp*
_output_shapes

:*
dtype0�
-model_1/prune_low_magnitude_fc3_pruned/MatMulMatMul)model_1/encoder_output/Relu:activations:0Dmodel_1/prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
=model_1/prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOpReadVariableOpFmodel_1_prune_low_magnitude_fc3_pruned_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
.model_1/prune_low_magnitude_fc3_pruned/BiasAddBiasAdd7model_1/prune_low_magnitude_fc3_pruned/MatMul:product:0Emodel_1/prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+model_1/prune_low_magnitude_fc3_pruned/ReluRelu7model_1/prune_low_magnitude_fc3_pruned/BiasAdd:output:0*
T0*'
_output_shapes
:���������Q
3model_1/prune_low_magnitude_fc3_prunclass/no_updateNoOp*
_output_shapes
 S
5model_1/prune_low_magnitude_fc3_prunclass/no_update_1NoOp*
_output_shapes
 �
<model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOpReadVariableOpEmodel_1_prune_low_magnitude_fc3_prunclass_mul_readvariableop_resource*
_output_shapes

:*
dtype0�
>model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1ReadVariableOpGmodel_1_prune_low_magnitude_fc3_prunclass_mul_readvariableop_1_resource*
_output_shapes

:*
dtype0�
-model_1/prune_low_magnitude_fc3_prunclass/MulMulDmodel_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp:value:0Fmodel_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
:model_1/prune_low_magnitude_fc3_prunclass/AssignVariableOpAssignVariableOpEmodel_1_prune_low_magnitude_fc3_prunclass_mul_readvariableop_resource1model_1/prune_low_magnitude_fc3_prunclass/Mul:z:0=^model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
4model_1/prune_low_magnitude_fc3_prunclass/group_depsNoOp;^model_1/prune_low_magnitude_fc3_prunclass/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
6model_1/prune_low_magnitude_fc3_prunclass/group_deps_1NoOp5^model_1/prune_low_magnitude_fc3_prunclass/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
?model_1/prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOpReadVariableOpEmodel_1_prune_low_magnitude_fc3_prunclass_mul_readvariableop_resource;^model_1/prune_low_magnitude_fc3_prunclass/AssignVariableOp*
_output_shapes

:*
dtype0�
0model_1/prune_low_magnitude_fc3_prunclass/MatMulMatMul)model_1/encoder_output/Relu:activations:0Gmodel_1/prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
@model_1/prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOpReadVariableOpImodel_1_prune_low_magnitude_fc3_prunclass_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
1model_1/prune_low_magnitude_fc3_prunclass/BiasAddBiasAdd:model_1/prune_low_magnitude_fc3_prunclass/MatMul:product:0Hmodel_1/prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.model_1/prune_low_magnitude_fc3_prunclass/ReluRelu:model_1/prune_low_magnitude_fc3_prunclass/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!model_1/fc4/MatMul/ReadVariableOpReadVariableOp*model_1_fc4_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
model_1/fc4/MatMulMatMul9model_1/prune_low_magnitude_fc3_pruned/Relu:activations:0)model_1/fc4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
"model_1/fc4/BiasAdd/ReadVariableOpReadVariableOp+model_1_fc4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/fc4/BiasAddBiasAddmodel_1/fc4/MatMul:product:0*model_1/fc4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� h
model_1/fc4/ReluRelumodel_1/fc4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'model_1/fc4_class/MatMul/ReadVariableOpReadVariableOp0model_1_fc4_class_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_1/fc4_class/MatMulMatMul<model_1/prune_low_magnitude_fc3_prunclass/Relu:activations:0/model_1/fc4_class/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
(model_1/fc4_class/BiasAdd/ReadVariableOpReadVariableOp1model_1_fc4_class_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_1/fc4_class/BiasAddBiasAdd"model_1/fc4_class/MatMul:product:00model_1/fc4_class/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(t
model_1/fc4_class/ReluRelu"model_1/fc4_class/BiasAdd:output:0*
T0*'
_output_shapes
:���������(�
#model_1/fc4.1/MatMul/ReadVariableOpReadVariableOp,model_1_fc4_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model_1/fc4.1/MatMulMatMulmodel_1/fc4/Relu:activations:0+model_1/fc4.1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
$model_1/fc4.1/BiasAdd/ReadVariableOpReadVariableOp-model_1_fc4_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/fc4.1/BiasAddBiasAddmodel_1/fc4.1/MatMul:product:0,model_1/fc4.1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� l
model_1/fc4.1/ReluRelumodel_1/fc4.1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
/model_1/classifier_output/MatMul/ReadVariableOpReadVariableOp8model_1_classifier_output_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
 model_1/classifier_output/MatMulMatMul$model_1/fc4_class/Relu:activations:07model_1/classifier_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
0model_1/classifier_output/BiasAdd/ReadVariableOpReadVariableOp9model_1_classifier_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
!model_1/classifier_output/BiasAddBiasAdd*model_1/classifier_output/MatMul:product:08model_1/classifier_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
!model_1/classifier_output/SoftmaxSoftmax*model_1/classifier_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
+model_1/ecoder_output/MatMul/ReadVariableOpReadVariableOp4model_1_ecoder_output_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
model_1/ecoder_output/MatMulMatMul model_1/fc4.1/Relu:activations:03model_1/ecoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,model_1/ecoder_output/BiasAdd/ReadVariableOpReadVariableOp5model_1_ecoder_output_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/ecoder_output/BiasAddBiasAdd&model_1/ecoder_output/MatMul:product:04model_1/ecoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
model_1/ecoder_output/SigmoidSigmoid&model_1/ecoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������@z
IdentityIdentity+model_1/classifier_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
r

Identity_1Identity!model_1/ecoder_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp1^model_1/classifier_output/BiasAdd/ReadVariableOp0^model_1/classifier_output/MatMul/ReadVariableOp-^model_1/ecoder_output/BiasAdd/ReadVariableOp,^model_1/ecoder_output/MatMul/ReadVariableOp.^model_1/encoder_output/BiasAdd/ReadVariableOp-^model_1/encoder_output/MatMul/ReadVariableOp#^model_1/fc1/BiasAdd/ReadVariableOp"^model_1/fc1/MatMul/ReadVariableOp%^model_1/fc4.1/BiasAdd/ReadVariableOp$^model_1/fc4.1/MatMul/ReadVariableOp#^model_1/fc4/BiasAdd/ReadVariableOp"^model_1/fc4/MatMul/ReadVariableOp)^model_1/fc4_class/BiasAdd/ReadVariableOp(^model_1/fc4_class/MatMul/ReadVariableOp8^model_1/prune_low_magnitude_fc2.1_prun/AssignVariableOp>^model_1/prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp=^model_1/prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp:^model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp<^model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_16^model_1/prune_low_magnitude_fc2_prun/AssignVariableOp<^model_1/prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp;^model_1/prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp8^model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp:^model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1;^model_1/prune_low_magnitude_fc3_prunclass/AssignVariableOpA^model_1/prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp@^model_1/prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp=^model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp?^model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_18^model_1/prune_low_magnitude_fc3_pruned/AssignVariableOp>^model_1/prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp=^model_1/prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp:^model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp<^model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0model_1/classifier_output/BiasAdd/ReadVariableOp0model_1/classifier_output/BiasAdd/ReadVariableOp2b
/model_1/classifier_output/MatMul/ReadVariableOp/model_1/classifier_output/MatMul/ReadVariableOp2\
,model_1/ecoder_output/BiasAdd/ReadVariableOp,model_1/ecoder_output/BiasAdd/ReadVariableOp2Z
+model_1/ecoder_output/MatMul/ReadVariableOp+model_1/ecoder_output/MatMul/ReadVariableOp2^
-model_1/encoder_output/BiasAdd/ReadVariableOp-model_1/encoder_output/BiasAdd/ReadVariableOp2\
,model_1/encoder_output/MatMul/ReadVariableOp,model_1/encoder_output/MatMul/ReadVariableOp2H
"model_1/fc1/BiasAdd/ReadVariableOp"model_1/fc1/BiasAdd/ReadVariableOp2F
!model_1/fc1/MatMul/ReadVariableOp!model_1/fc1/MatMul/ReadVariableOp2L
$model_1/fc4.1/BiasAdd/ReadVariableOp$model_1/fc4.1/BiasAdd/ReadVariableOp2J
#model_1/fc4.1/MatMul/ReadVariableOp#model_1/fc4.1/MatMul/ReadVariableOp2H
"model_1/fc4/BiasAdd/ReadVariableOp"model_1/fc4/BiasAdd/ReadVariableOp2F
!model_1/fc4/MatMul/ReadVariableOp!model_1/fc4/MatMul/ReadVariableOp2T
(model_1/fc4_class/BiasAdd/ReadVariableOp(model_1/fc4_class/BiasAdd/ReadVariableOp2R
'model_1/fc4_class/MatMul/ReadVariableOp'model_1/fc4_class/MatMul/ReadVariableOp2r
7model_1/prune_low_magnitude_fc2.1_prun/AssignVariableOp7model_1/prune_low_magnitude_fc2.1_prun/AssignVariableOp2~
=model_1/prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp=model_1/prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp2|
<model_1/prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp<model_1/prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp2v
9model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp9model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp2z
;model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1;model_1/prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_12n
5model_1/prune_low_magnitude_fc2_prun/AssignVariableOp5model_1/prune_low_magnitude_fc2_prun/AssignVariableOp2z
;model_1/prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp;model_1/prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp2x
:model_1/prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp:model_1/prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp2r
7model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp7model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2v
9model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_19model_1/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_12x
:model_1/prune_low_magnitude_fc3_prunclass/AssignVariableOp:model_1/prune_low_magnitude_fc3_prunclass/AssignVariableOp2�
@model_1/prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp@model_1/prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp2�
?model_1/prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp?model_1/prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp2|
<model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp<model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp2�
>model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1>model_1/prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_12r
7model_1/prune_low_magnitude_fc3_pruned/AssignVariableOp7model_1/prune_low_magnitude_fc3_pruned/AssignVariableOp2~
=model_1/prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp=model_1/prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp2|
<model_1/prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp<model_1/prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp2v
9model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp9model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp2z
;model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1;model_1/prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�
�
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_493258

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��AssignVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1'
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
 �
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�F
�
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_495918

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: -
biasadd_readvariableop_resource:
identity��AssignVariableOp�AssignVariableOp_1�BiasAdd/ReadVariableOp�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
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
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_495793*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_495792�
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
 *��Y?�
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
*$
else_branchR
cond_false_495833*
output_shapes
: *#
then_branchR
cond_true_495832q
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
 w
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_120
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�>
�
cond_true_4956463
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
 *��Y?�
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
cond_false_493888
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
�
�
&__inference_fc4.1_layer_call_fn_496339

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc4.1_layer_call_and_return_conditional_losses_493311o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
B__inference_prune_low_magnitude_fc3_prunclass_layer_call_fn_496170

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
GPU 2J 8� *f
faR_
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_493619o
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
�

�
E__inference_fc4_class_layer_call_and_return_conditional_losses_496370

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4assert_greater_equal_Assert_AssertGuard_false_493848K
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
A__inference_fc4.1_layer_call_and_return_conditional_losses_493311

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_495979

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��AssignVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1'
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
 �
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�>
�
cond_true_4938873
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
 *��Y?�
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
?__inference_fc1_layer_call_and_return_conditional_losses_495546

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
0prune_low_magnitude_fc3_pruned_cond_false_4952643
/prune_low_magnitude_fc3_pruned_cond_placeholder5
1prune_low_magnitude_fc3_pruned_cond_placeholder_15
1prune_low_magnitude_fc3_pruned_cond_placeholder_25
1prune_low_magnitude_fc3_pruned_cond_placeholder_3\
Xprune_low_magnitude_fc3_pruned_cond_identity_prune_low_magnitude_fc3_pruned_logicaland_1
2
.prune_low_magnitude_fc3_pruned_cond_identity_1
n
(prune_low_magnitude_fc3_pruned/cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
,prune_low_magnitude_fc3_pruned/cond/IdentityIdentityXprune_low_magnitude_fc3_pruned_cond_identity_prune_low_magnitude_fc3_pruned_logicaland_1)^prune_low_magnitude_fc3_pruned/cond/NoOp*
T0
*
_output_shapes
: �
.prune_low_magnitude_fc3_pruned/cond/Identity_1Identity5prune_low_magnitude_fc3_pruned/cond/Identity:output:0*
T0
*
_output_shapes
: "i
.prune_low_magnitude_fc3_pruned_cond_identity_17prune_low_magnitude_fc3_pruned/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
�
�
cond_false_493534
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
�
Rprune_low_magnitude_fc2.1_prun_assert_greater_equal_Assert_AssertGuard_true_495075�
�prune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc2_1_prun_assert_greater_equal_all
V
Rprune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_placeholder	X
Tprune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_placeholder_1	U
Qprune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_identity_1
�
Kprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Oprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc2_1_prun_assert_greater_equal_allL^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Qprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityXprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "�
Qprune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_identity_1Zprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
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
�
�
Sprune_low_magnitude_fc2.1_prun_assert_greater_equal_Assert_AssertGuard_false_495076�
�prune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_1_prun_assert_greater_equal_all
�
�prune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_1_prun_assert_greater_equal_readvariableop	�
�prune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_1_prun_assert_greater_equal_y	U
Qprune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_identity_1
��Mprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert�
Tprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Tprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Tprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*\
valueSBQ BKx (prune_low_magnitude_fc2.1_prun/assert_greater_equal/ReadVariableOp:0) = �
Tprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (prune_low_magnitude_fc2.1_prun/assert_greater_equal/y:0) = �
Mprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/AssertAssert�prune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_1_prun_assert_greater_equal_all]prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0]prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0]prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0�prune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_1_prun_assert_greater_equal_readvariableop]prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0�prune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_1_prun_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Oprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc2_1_prun_assert_greater_equal_allN^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
Qprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityXprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity:output:0L^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Kprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/NoOpNoOpN^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "�
Qprune_low_magnitude_fc2_1_prun_assert_greater_equal_assert_assertguard_identity_1Zprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2�
Mprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/AssertMprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Assert: 
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
�
Pprune_low_magnitude_fc2_prun_assert_greater_equal_Assert_AssertGuard_true_494934�
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
�

�
E__inference_fc4_class_layer_call_and_return_conditional_losses_493294

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
?__inference_fc1_layer_call_and_return_conditional_losses_493151

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
֒
�+
C__inference_model_1_layer_call_and_return_conditional_losses_495526

inputs4
"fc1_matmul_readvariableop_resource:@1
#fc1_biasadd_readvariableop_resource:>
4prune_low_magnitude_fc2_prun_readvariableop_resource:	 ;
)prune_low_magnitude_fc2_prun_cond_input_1:;
)prune_low_magnitude_fc2_prun_cond_input_2:3
)prune_low_magnitude_fc2_prun_cond_input_3: J
<prune_low_magnitude_fc2_prun_biasadd_readvariableop_resource:@
6prune_low_magnitude_fc2_1_prun_readvariableop_resource:	 =
+prune_low_magnitude_fc2_1_prun_cond_input_1:=
+prune_low_magnitude_fc2_1_prun_cond_input_2:5
+prune_low_magnitude_fc2_1_prun_cond_input_3: L
>prune_low_magnitude_fc2_1_prun_biasadd_readvariableop_resource:?
-encoder_output_matmul_readvariableop_resource:<
.encoder_output_biasadd_readvariableop_resource:@
6prune_low_magnitude_fc3_pruned_readvariableop_resource:	 =
+prune_low_magnitude_fc3_pruned_cond_input_1:=
+prune_low_magnitude_fc3_pruned_cond_input_2:5
+prune_low_magnitude_fc3_pruned_cond_input_3: L
>prune_low_magnitude_fc3_pruned_biasadd_readvariableop_resource:C
9prune_low_magnitude_fc3_prunclass_readvariableop_resource:	 @
.prune_low_magnitude_fc3_prunclass_cond_input_1:@
.prune_low_magnitude_fc3_prunclass_cond_input_2:8
.prune_low_magnitude_fc3_prunclass_cond_input_3: O
Aprune_low_magnitude_fc3_prunclass_biasadd_readvariableop_resource:4
"fc4_matmul_readvariableop_resource: 1
#fc4_biasadd_readvariableop_resource: :
(fc4_class_matmul_readvariableop_resource:(7
)fc4_class_biasadd_readvariableop_resource:(6
$fc4_1_matmul_readvariableop_resource:  3
%fc4_1_biasadd_readvariableop_resource: B
0classifier_output_matmul_readvariableop_resource:(
?
1classifier_output_biasadd_readvariableop_resource:
>
,ecoder_output_matmul_readvariableop_resource: @;
-ecoder_output_biasadd_readvariableop_resource:@
identity

identity_1��(classifier_output/BiasAdd/ReadVariableOp�'classifier_output/MatMul/ReadVariableOp�$ecoder_output/BiasAdd/ReadVariableOp�#ecoder_output/MatMul/ReadVariableOp�%encoder_output/BiasAdd/ReadVariableOp�$encoder_output/MatMul/ReadVariableOp�fc1/BiasAdd/ReadVariableOp�fc1/MatMul/ReadVariableOp�fc4.1/BiasAdd/ReadVariableOp�fc4.1/MatMul/ReadVariableOp�fc4/BiasAdd/ReadVariableOp�fc4/MatMul/ReadVariableOp� fc4_class/BiasAdd/ReadVariableOp�fc4_class/MatMul/ReadVariableOp�/prune_low_magnitude_fc2.1_prun/AssignVariableOp�1prune_low_magnitude_fc2.1_prun/AssignVariableOp_1�5prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp�:prune_low_magnitude_fc2.1_prun/GreaterEqual/ReadVariableOp�7prune_low_magnitude_fc2.1_prun/LessEqual/ReadVariableOp�4prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp�1prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp�3prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1�-prune_low_magnitude_fc2.1_prun/ReadVariableOp�1prune_low_magnitude_fc2.1_prun/Sub/ReadVariableOp�Fprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard�Bprune_low_magnitude_fc2.1_prun/assert_greater_equal/ReadVariableOp�#prune_low_magnitude_fc2.1_prun/cond�-prune_low_magnitude_fc2_prun/AssignVariableOp�/prune_low_magnitude_fc2_prun/AssignVariableOp_1�3prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp�8prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOp�5prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOp�2prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp�/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp�1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1�+prune_low_magnitude_fc2_prun/ReadVariableOp�/prune_low_magnitude_fc2_prun/Sub/ReadVariableOp�Dprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard�@prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp�!prune_low_magnitude_fc2_prun/cond�2prune_low_magnitude_fc3_prunclass/AssignVariableOp�4prune_low_magnitude_fc3_prunclass/AssignVariableOp_1�8prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp�=prune_low_magnitude_fc3_prunclass/GreaterEqual/ReadVariableOp�:prune_low_magnitude_fc3_prunclass/LessEqual/ReadVariableOp�7prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp�4prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp�6prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1�0prune_low_magnitude_fc3_prunclass/ReadVariableOp�4prune_low_magnitude_fc3_prunclass/Sub/ReadVariableOp�Iprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard�Eprune_low_magnitude_fc3_prunclass/assert_greater_equal/ReadVariableOp�&prune_low_magnitude_fc3_prunclass/cond�/prune_low_magnitude_fc3_pruned/AssignVariableOp�1prune_low_magnitude_fc3_pruned/AssignVariableOp_1�5prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp�:prune_low_magnitude_fc3_pruned/GreaterEqual/ReadVariableOp�7prune_low_magnitude_fc3_pruned/LessEqual/ReadVariableOp�4prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp�1prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp�3prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1�-prune_low_magnitude_fc3_pruned/ReadVariableOp�1prune_low_magnitude_fc3_pruned/Sub/ReadVariableOp�Fprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard�Bprune_low_magnitude_fc3_pruned/assert_greater_equal/ReadVariableOp�#prune_low_magnitude_fc3_pruned/cond|
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0q

fc1/MatMulMatMulinputs!fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������X
fc1/ReluRelufc1/BiasAdd:output:0*
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
 *d
else_branchURS
Qprune_low_magnitude_fc2_prun_assert_greater_equal_Assert_AssertGuard_false_494935*
output_shapes
: *c
then_branchTRR
Pprune_low_magnitude_fc2_prun_assert_greater_equal_Assert_AssertGuard_true_494934�
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
 *��Y?�
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
*A
else_branch2R0
.prune_low_magnitude_fc2_prun_cond_false_494975*
output_shapes
: *@
then_branch1R/
-prune_low_magnitude_fc2_prun_cond_true_494974�
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
 �
2prune_low_magnitude_fc2_prun/MatMul/ReadVariableOpReadVariableOp)prune_low_magnitude_fc2_prun_cond_input_10^prune_low_magnitude_fc2_prun/AssignVariableOp_1*
_output_shapes

:*
dtype0�
#prune_low_magnitude_fc2_prun/MatMulMatMulfc1/Relu:activations:0:prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
3prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOpReadVariableOp<prune_low_magnitude_fc2_prun_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$prune_low_magnitude_fc2_prun/BiasAddBiasAdd-prune_low_magnitude_fc2_prun/MatMul:product:0;prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!prune_low_magnitude_fc2_prun/ReluRelu-prune_low_magnitude_fc2_prun/BiasAdd:output:0*
T0*'
_output_shapes
:����������
-prune_low_magnitude_fc2.1_prun/ReadVariableOpReadVariableOp6prune_low_magnitude_fc2_1_prun_readvariableop_resource*
_output_shapes
: *
dtype0	f
$prune_low_magnitude_fc2.1_prun/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
"prune_low_magnitude_fc2.1_prun/addAddV25prune_low_magnitude_fc2.1_prun/ReadVariableOp:value:0-prune_low_magnitude_fc2.1_prun/add/y:output:0*
T0	*
_output_shapes
: �
/prune_low_magnitude_fc2.1_prun/AssignVariableOpAssignVariableOp6prune_low_magnitude_fc2_1_prun_readvariableop_resource&prune_low_magnitude_fc2.1_prun/add:z:0.^prune_low_magnitude_fc2.1_prun/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(u
%prune_low_magnitude_fc2.1_prun/updateNoOp0^prune_low_magnitude_fc2.1_prun/AssignVariableOp*
_output_shapes
 �
Bprune_low_magnitude_fc2.1_prun/assert_greater_equal/ReadVariableOpReadVariableOp6prune_low_magnitude_fc2_1_prun_readvariableop_resource0^prune_low_magnitude_fc2.1_prun/AssignVariableOp*
_output_shapes
: *
dtype0	w
5prune_low_magnitude_fc2.1_prun/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
@prune_low_magnitude_fc2.1_prun/assert_greater_equal/GreaterEqualGreaterEqualJprune_low_magnitude_fc2.1_prun/assert_greater_equal/ReadVariableOp:value:0>prune_low_magnitude_fc2.1_prun/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: z
8prune_low_magnitude_fc2.1_prun/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : �
?prune_low_magnitude_fc2.1_prun/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
?prune_low_magnitude_fc2.1_prun/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
9prune_low_magnitude_fc2.1_prun/assert_greater_equal/rangeRangeHprune_low_magnitude_fc2.1_prun/assert_greater_equal/range/start:output:0Aprune_low_magnitude_fc2.1_prun/assert_greater_equal/Rank:output:0Hprune_low_magnitude_fc2.1_prun/assert_greater_equal/range/delta:output:0*
_output_shapes
: �
7prune_low_magnitude_fc2.1_prun/assert_greater_equal/AllAllDprune_low_magnitude_fc2.1_prun/assert_greater_equal/GreaterEqual:z:0Bprune_low_magnitude_fc2.1_prun/assert_greater_equal/range:output:0*
_output_shapes
: �
@prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Bprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Bprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*\
valueSBQ BKx (prune_low_magnitude_fc2.1_prun/assert_greater_equal/ReadVariableOp:0) = �
Bprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (prune_low_magnitude_fc2.1_prun/assert_greater_equal/y:0) = �
Fprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuardIf@prune_low_magnitude_fc2.1_prun/assert_greater_equal/All:output:0@prune_low_magnitude_fc2.1_prun/assert_greater_equal/All:output:0Jprune_low_magnitude_fc2.1_prun/assert_greater_equal/ReadVariableOp:value:0>prune_low_magnitude_fc2.1_prun/assert_greater_equal/y:output:0E^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard*
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
 *f
else_branchWRU
Sprune_low_magnitude_fc2.1_prun_assert_greater_equal_Assert_AssertGuard_false_495076*
output_shapes
: *e
then_branchVRT
Rprune_low_magnitude_fc2.1_prun_assert_greater_equal_Assert_AssertGuard_true_495075�
Oprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/IdentityIdentityOprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
:prune_low_magnitude_fc2.1_prun/GreaterEqual/ReadVariableOpReadVariableOp6prune_low_magnitude_fc2_1_prun_readvariableop_resource0^prune_low_magnitude_fc2.1_prun/AssignVariableOpP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
-prune_low_magnitude_fc2.1_prun/GreaterEqual/yConstP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
+prune_low_magnitude_fc2.1_prun/GreaterEqualGreaterEqualBprune_low_magnitude_fc2.1_prun/GreaterEqual/ReadVariableOp:value:06prune_low_magnitude_fc2.1_prun/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
7prune_low_magnitude_fc2.1_prun/LessEqual/ReadVariableOpReadVariableOp6prune_low_magnitude_fc2_1_prun_readvariableop_resource0^prune_low_magnitude_fc2.1_prun/AssignVariableOpP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
*prune_low_magnitude_fc2.1_prun/LessEqual/yConstP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�N�
(prune_low_magnitude_fc2.1_prun/LessEqual	LessEqual?prune_low_magnitude_fc2.1_prun/LessEqual/ReadVariableOp:value:03prune_low_magnitude_fc2.1_prun/LessEqual/y:output:0*
T0	*
_output_shapes
: �
%prune_low_magnitude_fc2.1_prun/Less/xConstP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N�
%prune_low_magnitude_fc2.1_prun/Less/yConstP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : �
#prune_low_magnitude_fc2.1_prun/LessLess.prune_low_magnitude_fc2.1_prun/Less/x:output:0.prune_low_magnitude_fc2.1_prun/Less/y:output:0*
T0*
_output_shapes
: �
(prune_low_magnitude_fc2.1_prun/LogicalOr	LogicalOr,prune_low_magnitude_fc2.1_prun/LessEqual:z:0'prune_low_magnitude_fc2.1_prun/Less:z:0*
_output_shapes
: �
)prune_low_magnitude_fc2.1_prun/LogicalAnd
LogicalAnd/prune_low_magnitude_fc2.1_prun/GreaterEqual:z:0,prune_low_magnitude_fc2.1_prun/LogicalOr:z:0*
_output_shapes
: �
1prune_low_magnitude_fc2.1_prun/Sub/ReadVariableOpReadVariableOp6prune_low_magnitude_fc2_1_prun_readvariableop_resource0^prune_low_magnitude_fc2.1_prun/AssignVariableOpP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
$prune_low_magnitude_fc2.1_prun/Sub/yConstP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
"prune_low_magnitude_fc2.1_prun/SubSub9prune_low_magnitude_fc2.1_prun/Sub/ReadVariableOp:value:0-prune_low_magnitude_fc2.1_prun/Sub/y:output:0*
T0	*
_output_shapes
: �
)prune_low_magnitude_fc2.1_prun/FloorMod/yConstP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd�
'prune_low_magnitude_fc2.1_prun/FloorModFloorMod&prune_low_magnitude_fc2.1_prun/Sub:z:02prune_low_magnitude_fc2.1_prun/FloorMod/y:output:0*
T0	*
_output_shapes
: �
&prune_low_magnitude_fc2.1_prun/Equal/yConstP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R �
$prune_low_magnitude_fc2.1_prun/EqualEqual+prune_low_magnitude_fc2.1_prun/FloorMod:z:0/prune_low_magnitude_fc2.1_prun/Equal/y:output:0*
T0	*
_output_shapes
: �
+prune_low_magnitude_fc2.1_prun/LogicalAnd_1
LogicalAnd-prune_low_magnitude_fc2.1_prun/LogicalAnd:z:0(prune_low_magnitude_fc2.1_prun/Equal:z:0*
_output_shapes
: �
$prune_low_magnitude_fc2.1_prun/ConstConstP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��Y?�
#prune_low_magnitude_fc2.1_prun/condIf/prune_low_magnitude_fc2.1_prun/LogicalAnd_1:z:06prune_low_magnitude_fc2_1_prun_readvariableop_resource+prune_low_magnitude_fc2_1_prun_cond_input_1+prune_low_magnitude_fc2_1_prun_cond_input_2+prune_low_magnitude_fc2_1_prun_cond_input_3/prune_low_magnitude_fc2.1_prun/LogicalAnd_1:z:00^prune_low_magnitude_fc2.1_prun/AssignVariableOp*
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
*C
else_branch4R2
0prune_low_magnitude_fc2.1_prun_cond_false_495116*
output_shapes
: *B
then_branch3R1
/prune_low_magnitude_fc2.1_prun_cond_true_495115�
,prune_low_magnitude_fc2.1_prun/cond/IdentityIdentity,prune_low_magnitude_fc2.1_prun/cond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
'prune_low_magnitude_fc2.1_prun/update_1NoOpP^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard/Identity-^prune_low_magnitude_fc2.1_prun/cond/Identity*
_output_shapes
 �
1prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOpReadVariableOp+prune_low_magnitude_fc2_1_prun_cond_input_1*
_output_shapes

:*
dtype0�
3prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1ReadVariableOp+prune_low_magnitude_fc2_1_prun_cond_input_2$^prune_low_magnitude_fc2.1_prun/cond*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc2.1_prun/MulMul9prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp:value:0;prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
1prune_low_magnitude_fc2.1_prun/AssignVariableOp_1AssignVariableOp+prune_low_magnitude_fc2_1_prun_cond_input_1&prune_low_magnitude_fc2.1_prun/Mul:z:02^prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp$^prune_low_magnitude_fc2.1_prun/cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
)prune_low_magnitude_fc2.1_prun/group_depsNoOp2^prune_low_magnitude_fc2.1_prun/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
+prune_low_magnitude_fc2.1_prun/group_deps_1NoOp*^prune_low_magnitude_fc2.1_prun/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
4prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOpReadVariableOp+prune_low_magnitude_fc2_1_prun_cond_input_12^prune_low_magnitude_fc2.1_prun/AssignVariableOp_1*
_output_shapes

:*
dtype0�
%prune_low_magnitude_fc2.1_prun/MatMulMatMul/prune_low_magnitude_fc2_prun/Relu:activations:0<prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOpReadVariableOp>prune_low_magnitude_fc2_1_prun_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&prune_low_magnitude_fc2.1_prun/BiasAddBiasAdd/prune_low_magnitude_fc2.1_prun/MatMul:product:0=prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#prune_low_magnitude_fc2.1_prun/ReluRelu/prune_low_magnitude_fc2.1_prun/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$encoder_output/MatMul/ReadVariableOpReadVariableOp-encoder_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
encoder_output/MatMulMatMul1prune_low_magnitude_fc2.1_prun/Relu:activations:0,encoder_output/MatMul/ReadVariableOp:value:0*
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
-prune_low_magnitude_fc3_pruned/ReadVariableOpReadVariableOp6prune_low_magnitude_fc3_pruned_readvariableop_resource*
_output_shapes
: *
dtype0	f
$prune_low_magnitude_fc3_pruned/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
"prune_low_magnitude_fc3_pruned/addAddV25prune_low_magnitude_fc3_pruned/ReadVariableOp:value:0-prune_low_magnitude_fc3_pruned/add/y:output:0*
T0	*
_output_shapes
: �
/prune_low_magnitude_fc3_pruned/AssignVariableOpAssignVariableOp6prune_low_magnitude_fc3_pruned_readvariableop_resource&prune_low_magnitude_fc3_pruned/add:z:0.^prune_low_magnitude_fc3_pruned/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape(u
%prune_low_magnitude_fc3_pruned/updateNoOp0^prune_low_magnitude_fc3_pruned/AssignVariableOp*
_output_shapes
 �
Bprune_low_magnitude_fc3_pruned/assert_greater_equal/ReadVariableOpReadVariableOp6prune_low_magnitude_fc3_pruned_readvariableop_resource0^prune_low_magnitude_fc3_pruned/AssignVariableOp*
_output_shapes
: *
dtype0	w
5prune_low_magnitude_fc3_pruned/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
@prune_low_magnitude_fc3_pruned/assert_greater_equal/GreaterEqualGreaterEqualJprune_low_magnitude_fc3_pruned/assert_greater_equal/ReadVariableOp:value:0>prune_low_magnitude_fc3_pruned/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: z
8prune_low_magnitude_fc3_pruned/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : �
?prune_low_magnitude_fc3_pruned/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
?prune_low_magnitude_fc3_pruned/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
9prune_low_magnitude_fc3_pruned/assert_greater_equal/rangeRangeHprune_low_magnitude_fc3_pruned/assert_greater_equal/range/start:output:0Aprune_low_magnitude_fc3_pruned/assert_greater_equal/Rank:output:0Hprune_low_magnitude_fc3_pruned/assert_greater_equal/range/delta:output:0*
_output_shapes
: �
7prune_low_magnitude_fc3_pruned/assert_greater_equal/AllAllDprune_low_magnitude_fc3_pruned/assert_greater_equal/GreaterEqual:z:0Bprune_low_magnitude_fc3_pruned/assert_greater_equal/range:output:0*
_output_shapes
: �
@prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Bprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Bprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*\
valueSBQ BKx (prune_low_magnitude_fc3_pruned/assert_greater_equal/ReadVariableOp:0) = �
Bprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (prune_low_magnitude_fc3_pruned/assert_greater_equal/y:0) = �
Fprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuardIf@prune_low_magnitude_fc3_pruned/assert_greater_equal/All:output:0@prune_low_magnitude_fc3_pruned/assert_greater_equal/All:output:0Jprune_low_magnitude_fc3_pruned/assert_greater_equal/ReadVariableOp:value:0>prune_low_magnitude_fc3_pruned/assert_greater_equal/y:output:0G^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard*
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
 *f
else_branchWRU
Sprune_low_magnitude_fc3_pruned_assert_greater_equal_Assert_AssertGuard_false_495224*
output_shapes
: *e
then_branchVRT
Rprune_low_magnitude_fc3_pruned_assert_greater_equal_Assert_AssertGuard_true_495223�
Oprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/IdentityIdentityOprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
:prune_low_magnitude_fc3_pruned/GreaterEqual/ReadVariableOpReadVariableOp6prune_low_magnitude_fc3_pruned_readvariableop_resource0^prune_low_magnitude_fc3_pruned/AssignVariableOpP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
-prune_low_magnitude_fc3_pruned/GreaterEqual/yConstP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
+prune_low_magnitude_fc3_pruned/GreaterEqualGreaterEqualBprune_low_magnitude_fc3_pruned/GreaterEqual/ReadVariableOp:value:06prune_low_magnitude_fc3_pruned/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
7prune_low_magnitude_fc3_pruned/LessEqual/ReadVariableOpReadVariableOp6prune_low_magnitude_fc3_pruned_readvariableop_resource0^prune_low_magnitude_fc3_pruned/AssignVariableOpP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
*prune_low_magnitude_fc3_pruned/LessEqual/yConstP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�N�
(prune_low_magnitude_fc3_pruned/LessEqual	LessEqual?prune_low_magnitude_fc3_pruned/LessEqual/ReadVariableOp:value:03prune_low_magnitude_fc3_pruned/LessEqual/y:output:0*
T0	*
_output_shapes
: �
%prune_low_magnitude_fc3_pruned/Less/xConstP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N�
%prune_low_magnitude_fc3_pruned/Less/yConstP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : �
#prune_low_magnitude_fc3_pruned/LessLess.prune_low_magnitude_fc3_pruned/Less/x:output:0.prune_low_magnitude_fc3_pruned/Less/y:output:0*
T0*
_output_shapes
: �
(prune_low_magnitude_fc3_pruned/LogicalOr	LogicalOr,prune_low_magnitude_fc3_pruned/LessEqual:z:0'prune_low_magnitude_fc3_pruned/Less:z:0*
_output_shapes
: �
)prune_low_magnitude_fc3_pruned/LogicalAnd
LogicalAnd/prune_low_magnitude_fc3_pruned/GreaterEqual:z:0,prune_low_magnitude_fc3_pruned/LogicalOr:z:0*
_output_shapes
: �
1prune_low_magnitude_fc3_pruned/Sub/ReadVariableOpReadVariableOp6prune_low_magnitude_fc3_pruned_readvariableop_resource0^prune_low_magnitude_fc3_pruned/AssignVariableOpP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
$prune_low_magnitude_fc3_pruned/Sub/yConstP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
"prune_low_magnitude_fc3_pruned/SubSub9prune_low_magnitude_fc3_pruned/Sub/ReadVariableOp:value:0-prune_low_magnitude_fc3_pruned/Sub/y:output:0*
T0	*
_output_shapes
: �
)prune_low_magnitude_fc3_pruned/FloorMod/yConstP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd�
'prune_low_magnitude_fc3_pruned/FloorModFloorMod&prune_low_magnitude_fc3_pruned/Sub:z:02prune_low_magnitude_fc3_pruned/FloorMod/y:output:0*
T0	*
_output_shapes
: �
&prune_low_magnitude_fc3_pruned/Equal/yConstP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R �
$prune_low_magnitude_fc3_pruned/EqualEqual+prune_low_magnitude_fc3_pruned/FloorMod:z:0/prune_low_magnitude_fc3_pruned/Equal/y:output:0*
T0	*
_output_shapes
: �
+prune_low_magnitude_fc3_pruned/LogicalAnd_1
LogicalAnd-prune_low_magnitude_fc3_pruned/LogicalAnd:z:0(prune_low_magnitude_fc3_pruned/Equal:z:0*
_output_shapes
: �
$prune_low_magnitude_fc3_pruned/ConstConstP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��Y?�
#prune_low_magnitude_fc3_pruned/condIf/prune_low_magnitude_fc3_pruned/LogicalAnd_1:z:06prune_low_magnitude_fc3_pruned_readvariableop_resource+prune_low_magnitude_fc3_pruned_cond_input_1+prune_low_magnitude_fc3_pruned_cond_input_2+prune_low_magnitude_fc3_pruned_cond_input_3/prune_low_magnitude_fc3_pruned/LogicalAnd_1:z:00^prune_low_magnitude_fc3_pruned/AssignVariableOp*
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
*C
else_branch4R2
0prune_low_magnitude_fc3_pruned_cond_false_495264*
output_shapes
: *B
then_branch3R1
/prune_low_magnitude_fc3_pruned_cond_true_495263�
,prune_low_magnitude_fc3_pruned/cond/IdentityIdentity,prune_low_magnitude_fc3_pruned/cond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
'prune_low_magnitude_fc3_pruned/update_1NoOpP^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity-^prune_low_magnitude_fc3_pruned/cond/Identity*
_output_shapes
 �
1prune_low_magnitude_fc3_pruned/Mul/ReadVariableOpReadVariableOp+prune_low_magnitude_fc3_pruned_cond_input_1*
_output_shapes

:*
dtype0�
3prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1ReadVariableOp+prune_low_magnitude_fc3_pruned_cond_input_2$^prune_low_magnitude_fc3_pruned/cond*
_output_shapes

:*
dtype0�
"prune_low_magnitude_fc3_pruned/MulMul9prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp:value:0;prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
1prune_low_magnitude_fc3_pruned/AssignVariableOp_1AssignVariableOp+prune_low_magnitude_fc3_pruned_cond_input_1&prune_low_magnitude_fc3_pruned/Mul:z:02^prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp$^prune_low_magnitude_fc3_pruned/cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
)prune_low_magnitude_fc3_pruned/group_depsNoOp2^prune_low_magnitude_fc3_pruned/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
+prune_low_magnitude_fc3_pruned/group_deps_1NoOp*^prune_low_magnitude_fc3_pruned/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
4prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOpReadVariableOp+prune_low_magnitude_fc3_pruned_cond_input_12^prune_low_magnitude_fc3_pruned/AssignVariableOp_1*
_output_shapes

:*
dtype0�
%prune_low_magnitude_fc3_pruned/MatMulMatMul!encoder_output/Relu:activations:0<prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOpReadVariableOp>prune_low_magnitude_fc3_pruned_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&prune_low_magnitude_fc3_pruned/BiasAddBiasAdd/prune_low_magnitude_fc3_pruned/MatMul:product:0=prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#prune_low_magnitude_fc3_pruned/ReluRelu/prune_low_magnitude_fc3_pruned/BiasAdd:output:0*
T0*'
_output_shapes
:����������
0prune_low_magnitude_fc3_prunclass/ReadVariableOpReadVariableOp9prune_low_magnitude_fc3_prunclass_readvariableop_resource*
_output_shapes
: *
dtype0	i
'prune_low_magnitude_fc3_prunclass/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
%prune_low_magnitude_fc3_prunclass/addAddV28prune_low_magnitude_fc3_prunclass/ReadVariableOp:value:00prune_low_magnitude_fc3_prunclass/add/y:output:0*
T0	*
_output_shapes
: �
2prune_low_magnitude_fc3_prunclass/AssignVariableOpAssignVariableOp9prune_low_magnitude_fc3_prunclass_readvariableop_resource)prune_low_magnitude_fc3_prunclass/add:z:01^prune_low_magnitude_fc3_prunclass/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	*
validate_shape({
(prune_low_magnitude_fc3_prunclass/updateNoOp3^prune_low_magnitude_fc3_prunclass/AssignVariableOp*
_output_shapes
 �
Eprune_low_magnitude_fc3_prunclass/assert_greater_equal/ReadVariableOpReadVariableOp9prune_low_magnitude_fc3_prunclass_readvariableop_resource3^prune_low_magnitude_fc3_prunclass/AssignVariableOp*
_output_shapes
: *
dtype0	z
8prune_low_magnitude_fc3_prunclass/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
Cprune_low_magnitude_fc3_prunclass/assert_greater_equal/GreaterEqualGreaterEqualMprune_low_magnitude_fc3_prunclass/assert_greater_equal/ReadVariableOp:value:0Aprune_low_magnitude_fc3_prunclass/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: }
;prune_low_magnitude_fc3_prunclass/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Bprune_low_magnitude_fc3_prunclass/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Bprune_low_magnitude_fc3_prunclass/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
<prune_low_magnitude_fc3_prunclass/assert_greater_equal/rangeRangeKprune_low_magnitude_fc3_prunclass/assert_greater_equal/range/start:output:0Dprune_low_magnitude_fc3_prunclass/assert_greater_equal/Rank:output:0Kprune_low_magnitude_fc3_prunclass/assert_greater_equal/range/delta:output:0*
_output_shapes
: �
:prune_low_magnitude_fc3_prunclass/assert_greater_equal/AllAllGprune_low_magnitude_fc3_prunclass/assert_greater_equal/GreaterEqual:z:0Eprune_low_magnitude_fc3_prunclass/assert_greater_equal/range:output:0*
_output_shapes
: �
Cprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Eprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Eprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*_
valueVBT BNx (prune_low_magnitude_fc3_prunclass/assert_greater_equal/ReadVariableOp:0) = �
Eprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*R
valueIBG BAy (prune_low_magnitude_fc3_prunclass/assert_greater_equal/y:0) = �
Iprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuardIfCprune_low_magnitude_fc3_prunclass/assert_greater_equal/All:output:0Cprune_low_magnitude_fc3_prunclass/assert_greater_equal/All:output:0Mprune_low_magnitude_fc3_prunclass/assert_greater_equal/ReadVariableOp:value:0Aprune_low_magnitude_fc3_prunclass/assert_greater_equal/y:output:0G^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard*
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
 *i
else_branchZRX
Vprune_low_magnitude_fc3_prunclass_assert_greater_equal_Assert_AssertGuard_false_495365*
output_shapes
: *h
then_branchYRW
Uprune_low_magnitude_fc3_prunclass_assert_greater_equal_Assert_AssertGuard_true_495364�
Rprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/IdentityIdentityRprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
=prune_low_magnitude_fc3_prunclass/GreaterEqual/ReadVariableOpReadVariableOp9prune_low_magnitude_fc3_prunclass_readvariableop_resource3^prune_low_magnitude_fc3_prunclass/AssignVariableOpS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
0prune_low_magnitude_fc3_prunclass/GreaterEqual/yConstS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
.prune_low_magnitude_fc3_prunclass/GreaterEqualGreaterEqualEprune_low_magnitude_fc3_prunclass/GreaterEqual/ReadVariableOp:value:09prune_low_magnitude_fc3_prunclass/GreaterEqual/y:output:0*
T0	*
_output_shapes
: �
:prune_low_magnitude_fc3_prunclass/LessEqual/ReadVariableOpReadVariableOp9prune_low_magnitude_fc3_prunclass_readvariableop_resource3^prune_low_magnitude_fc3_prunclass/AssignVariableOpS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
-prune_low_magnitude_fc3_prunclass/LessEqual/yConstS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R�N�
+prune_low_magnitude_fc3_prunclass/LessEqual	LessEqualBprune_low_magnitude_fc3_prunclass/LessEqual/ReadVariableOp:value:06prune_low_magnitude_fc3_prunclass/LessEqual/y:output:0*
T0	*
_output_shapes
: �
(prune_low_magnitude_fc3_prunclass/Less/xConstS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value
B :�N�
(prune_low_magnitude_fc3_prunclass/Less/yConstS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : �
&prune_low_magnitude_fc3_prunclass/LessLess1prune_low_magnitude_fc3_prunclass/Less/x:output:01prune_low_magnitude_fc3_prunclass/Less/y:output:0*
T0*
_output_shapes
: �
+prune_low_magnitude_fc3_prunclass/LogicalOr	LogicalOr/prune_low_magnitude_fc3_prunclass/LessEqual:z:0*prune_low_magnitude_fc3_prunclass/Less:z:0*
_output_shapes
: �
,prune_low_magnitude_fc3_prunclass/LogicalAnd
LogicalAnd2prune_low_magnitude_fc3_prunclass/GreaterEqual:z:0/prune_low_magnitude_fc3_prunclass/LogicalOr:z:0*
_output_shapes
: �
4prune_low_magnitude_fc3_prunclass/Sub/ReadVariableOpReadVariableOp9prune_low_magnitude_fc3_prunclass_readvariableop_resource3^prune_low_magnitude_fc3_prunclass/AssignVariableOpS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	�
'prune_low_magnitude_fc3_prunclass/Sub/yConstS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value
B	 R��
%prune_low_magnitude_fc3_prunclass/SubSub<prune_low_magnitude_fc3_prunclass/Sub/ReadVariableOp:value:00prune_low_magnitude_fc3_prunclass/Sub/y:output:0*
T0	*
_output_shapes
: �
,prune_low_magnitude_fc3_prunclass/FloorMod/yConstS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd�
*prune_low_magnitude_fc3_prunclass/FloorModFloorMod)prune_low_magnitude_fc3_prunclass/Sub:z:05prune_low_magnitude_fc3_prunclass/FloorMod/y:output:0*
T0	*
_output_shapes
: �
)prune_low_magnitude_fc3_prunclass/Equal/yConstS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R �
'prune_low_magnitude_fc3_prunclass/EqualEqual.prune_low_magnitude_fc3_prunclass/FloorMod:z:02prune_low_magnitude_fc3_prunclass/Equal/y:output:0*
T0	*
_output_shapes
: �
.prune_low_magnitude_fc3_prunclass/LogicalAnd_1
LogicalAnd0prune_low_magnitude_fc3_prunclass/LogicalAnd:z:0+prune_low_magnitude_fc3_prunclass/Equal:z:0*
_output_shapes
: �
'prune_low_magnitude_fc3_prunclass/ConstConstS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *��Y?�
&prune_low_magnitude_fc3_prunclass/condIf2prune_low_magnitude_fc3_prunclass/LogicalAnd_1:z:09prune_low_magnitude_fc3_prunclass_readvariableop_resource.prune_low_magnitude_fc3_prunclass_cond_input_1.prune_low_magnitude_fc3_prunclass_cond_input_2.prune_low_magnitude_fc3_prunclass_cond_input_32prune_low_magnitude_fc3_prunclass/LogicalAnd_1:z:03^prune_low_magnitude_fc3_prunclass/AssignVariableOp*
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
*F
else_branch7R5
3prune_low_magnitude_fc3_prunclass_cond_false_495405*
output_shapes
: *E
then_branch6R4
2prune_low_magnitude_fc3_prunclass_cond_true_495404�
/prune_low_magnitude_fc3_prunclass/cond/IdentityIdentity/prune_low_magnitude_fc3_prunclass/cond:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: �
*prune_low_magnitude_fc3_prunclass/update_1NoOpS^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity0^prune_low_magnitude_fc3_prunclass/cond/Identity*
_output_shapes
 �
4prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOpReadVariableOp.prune_low_magnitude_fc3_prunclass_cond_input_1*
_output_shapes

:*
dtype0�
6prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1ReadVariableOp.prune_low_magnitude_fc3_prunclass_cond_input_2'^prune_low_magnitude_fc3_prunclass/cond*
_output_shapes

:*
dtype0�
%prune_low_magnitude_fc3_prunclass/MulMul<prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp:value:0>prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
4prune_low_magnitude_fc3_prunclass/AssignVariableOp_1AssignVariableOp.prune_low_magnitude_fc3_prunclass_cond_input_1)prune_low_magnitude_fc3_prunclass/Mul:z:05^prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp'^prune_low_magnitude_fc3_prunclass/cond*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
,prune_low_magnitude_fc3_prunclass/group_depsNoOp5^prune_low_magnitude_fc3_prunclass/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
.prune_low_magnitude_fc3_prunclass/group_deps_1NoOp-^prune_low_magnitude_fc3_prunclass/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 �
7prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOpReadVariableOp.prune_low_magnitude_fc3_prunclass_cond_input_15^prune_low_magnitude_fc3_prunclass/AssignVariableOp_1*
_output_shapes

:*
dtype0�
(prune_low_magnitude_fc3_prunclass/MatMulMatMul!encoder_output/Relu:activations:0?prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
8prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOpReadVariableOpAprune_low_magnitude_fc3_prunclass_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)prune_low_magnitude_fc3_prunclass/BiasAddBiasAdd2prune_low_magnitude_fc3_prunclass/MatMul:product:0@prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&prune_low_magnitude_fc3_prunclass/ReluRelu2prune_low_magnitude_fc3_prunclass/BiasAdd:output:0*
T0*'
_output_shapes
:���������|
fc4/MatMul/ReadVariableOpReadVariableOp"fc4_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�

fc4/MatMulMatMul1prune_low_magnitude_fc3_pruned/Relu:activations:0!fc4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� z
fc4/BiasAdd/ReadVariableOpReadVariableOp#fc4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
fc4/BiasAddBiasAddfc4/MatMul:product:0"fc4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� X
fc4/ReluRelufc4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
fc4_class/MatMul/ReadVariableOpReadVariableOp(fc4_class_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
fc4_class/MatMulMatMul4prune_low_magnitude_fc3_prunclass/Relu:activations:0'fc4_class/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
 fc4_class/BiasAdd/ReadVariableOpReadVariableOp)fc4_class_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
fc4_class/BiasAddBiasAddfc4_class/MatMul:product:0(fc4_class/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(d
fc4_class/ReluRelufc4_class/BiasAdd:output:0*
T0*'
_output_shapes
:���������(�
fc4.1/MatMul/ReadVariableOpReadVariableOp$fc4_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
fc4.1/MatMulMatMulfc4/Relu:activations:0#fc4.1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� ~
fc4.1/BiasAdd/ReadVariableOpReadVariableOp%fc4_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
fc4.1/BiasAddBiasAddfc4.1/MatMul:product:0$fc4.1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� \

fc4.1/ReluRelufc4.1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'classifier_output/MatMul/ReadVariableOpReadVariableOp0classifier_output_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
classifier_output/MatMulMatMulfc4_class/Relu:activations:0/classifier_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
(classifier_output/BiasAdd/ReadVariableOpReadVariableOp1classifier_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
classifier_output/BiasAddBiasAdd"classifier_output/MatMul:product:00classifier_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
z
classifier_output/SoftmaxSoftmax"classifier_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
#ecoder_output/MatMul/ReadVariableOpReadVariableOp,ecoder_output_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
ecoder_output/MatMulMatMulfc4.1/Relu:activations:0+ecoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$ecoder_output/BiasAdd/ReadVariableOpReadVariableOp-ecoder_output_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
ecoder_output/BiasAddBiasAddecoder_output/MatMul:product:0,ecoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
ecoder_output/SigmoidSigmoidecoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������@h
IdentityIdentityecoder_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@t

Identity_1Identity#classifier_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp)^classifier_output/BiasAdd/ReadVariableOp(^classifier_output/MatMul/ReadVariableOp%^ecoder_output/BiasAdd/ReadVariableOp$^ecoder_output/MatMul/ReadVariableOp&^encoder_output/BiasAdd/ReadVariableOp%^encoder_output/MatMul/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc4.1/BiasAdd/ReadVariableOp^fc4.1/MatMul/ReadVariableOp^fc4/BiasAdd/ReadVariableOp^fc4/MatMul/ReadVariableOp!^fc4_class/BiasAdd/ReadVariableOp ^fc4_class/MatMul/ReadVariableOp0^prune_low_magnitude_fc2.1_prun/AssignVariableOp2^prune_low_magnitude_fc2.1_prun/AssignVariableOp_16^prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp;^prune_low_magnitude_fc2.1_prun/GreaterEqual/ReadVariableOp8^prune_low_magnitude_fc2.1_prun/LessEqual/ReadVariableOp5^prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp2^prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp4^prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_1.^prune_low_magnitude_fc2.1_prun/ReadVariableOp2^prune_low_magnitude_fc2.1_prun/Sub/ReadVariableOpG^prune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuardC^prune_low_magnitude_fc2.1_prun/assert_greater_equal/ReadVariableOp$^prune_low_magnitude_fc2.1_prun/cond.^prune_low_magnitude_fc2_prun/AssignVariableOp0^prune_low_magnitude_fc2_prun/AssignVariableOp_14^prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp9^prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOp6^prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOp3^prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp0^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2^prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_1,^prune_low_magnitude_fc2_prun/ReadVariableOp0^prune_low_magnitude_fc2_prun/Sub/ReadVariableOpE^prune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuardA^prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp"^prune_low_magnitude_fc2_prun/cond3^prune_low_magnitude_fc3_prunclass/AssignVariableOp5^prune_low_magnitude_fc3_prunclass/AssignVariableOp_19^prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp>^prune_low_magnitude_fc3_prunclass/GreaterEqual/ReadVariableOp;^prune_low_magnitude_fc3_prunclass/LessEqual/ReadVariableOp8^prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp5^prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp7^prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_11^prune_low_magnitude_fc3_prunclass/ReadVariableOp5^prune_low_magnitude_fc3_prunclass/Sub/ReadVariableOpJ^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuardF^prune_low_magnitude_fc3_prunclass/assert_greater_equal/ReadVariableOp'^prune_low_magnitude_fc3_prunclass/cond0^prune_low_magnitude_fc3_pruned/AssignVariableOp2^prune_low_magnitude_fc3_pruned/AssignVariableOp_16^prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp;^prune_low_magnitude_fc3_pruned/GreaterEqual/ReadVariableOp8^prune_low_magnitude_fc3_pruned/LessEqual/ReadVariableOp5^prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp2^prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp4^prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_1.^prune_low_magnitude_fc3_pruned/ReadVariableOp2^prune_low_magnitude_fc3_pruned/Sub/ReadVariableOpG^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuardC^prune_low_magnitude_fc3_pruned/assert_greater_equal/ReadVariableOp$^prune_low_magnitude_fc3_pruned/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(classifier_output/BiasAdd/ReadVariableOp(classifier_output/BiasAdd/ReadVariableOp2R
'classifier_output/MatMul/ReadVariableOp'classifier_output/MatMul/ReadVariableOp2L
$ecoder_output/BiasAdd/ReadVariableOp$ecoder_output/BiasAdd/ReadVariableOp2J
#ecoder_output/MatMul/ReadVariableOp#ecoder_output/MatMul/ReadVariableOp2N
%encoder_output/BiasAdd/ReadVariableOp%encoder_output/BiasAdd/ReadVariableOp2L
$encoder_output/MatMul/ReadVariableOp$encoder_output/MatMul/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp2<
fc4.1/BiasAdd/ReadVariableOpfc4.1/BiasAdd/ReadVariableOp2:
fc4.1/MatMul/ReadVariableOpfc4.1/MatMul/ReadVariableOp28
fc4/BiasAdd/ReadVariableOpfc4/BiasAdd/ReadVariableOp26
fc4/MatMul/ReadVariableOpfc4/MatMul/ReadVariableOp2D
 fc4_class/BiasAdd/ReadVariableOp fc4_class/BiasAdd/ReadVariableOp2B
fc4_class/MatMul/ReadVariableOpfc4_class/MatMul/ReadVariableOp2b
/prune_low_magnitude_fc2.1_prun/AssignVariableOp/prune_low_magnitude_fc2.1_prun/AssignVariableOp2f
1prune_low_magnitude_fc2.1_prun/AssignVariableOp_11prune_low_magnitude_fc2.1_prun/AssignVariableOp_12n
5prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp5prune_low_magnitude_fc2.1_prun/BiasAdd/ReadVariableOp2x
:prune_low_magnitude_fc2.1_prun/GreaterEqual/ReadVariableOp:prune_low_magnitude_fc2.1_prun/GreaterEqual/ReadVariableOp2r
7prune_low_magnitude_fc2.1_prun/LessEqual/ReadVariableOp7prune_low_magnitude_fc2.1_prun/LessEqual/ReadVariableOp2l
4prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp4prune_low_magnitude_fc2.1_prun/MatMul/ReadVariableOp2f
1prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp1prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp2j
3prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_13prune_low_magnitude_fc2.1_prun/Mul/ReadVariableOp_12^
-prune_low_magnitude_fc2.1_prun/ReadVariableOp-prune_low_magnitude_fc2.1_prun/ReadVariableOp2f
1prune_low_magnitude_fc2.1_prun/Sub/ReadVariableOp1prune_low_magnitude_fc2.1_prun/Sub/ReadVariableOp2�
Fprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuardFprune_low_magnitude_fc2.1_prun/assert_greater_equal/Assert/AssertGuard2�
Bprune_low_magnitude_fc2.1_prun/assert_greater_equal/ReadVariableOpBprune_low_magnitude_fc2.1_prun/assert_greater_equal/ReadVariableOp2J
#prune_low_magnitude_fc2.1_prun/cond#prune_low_magnitude_fc2.1_prun/cond2^
-prune_low_magnitude_fc2_prun/AssignVariableOp-prune_low_magnitude_fc2_prun/AssignVariableOp2b
/prune_low_magnitude_fc2_prun/AssignVariableOp_1/prune_low_magnitude_fc2_prun/AssignVariableOp_12j
3prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp3prune_low_magnitude_fc2_prun/BiasAdd/ReadVariableOp2t
8prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOp8prune_low_magnitude_fc2_prun/GreaterEqual/ReadVariableOp2n
5prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOp5prune_low_magnitude_fc2_prun/LessEqual/ReadVariableOp2h
2prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp2prune_low_magnitude_fc2_prun/MatMul/ReadVariableOp2b
/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp/prune_low_magnitude_fc2_prun/Mul/ReadVariableOp2f
1prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_11prune_low_magnitude_fc2_prun/Mul/ReadVariableOp_12Z
+prune_low_magnitude_fc2_prun/ReadVariableOp+prune_low_magnitude_fc2_prun/ReadVariableOp2b
/prune_low_magnitude_fc2_prun/Sub/ReadVariableOp/prune_low_magnitude_fc2_prun/Sub/ReadVariableOp2�
Dprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuardDprune_low_magnitude_fc2_prun/assert_greater_equal/Assert/AssertGuard2�
@prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp@prune_low_magnitude_fc2_prun/assert_greater_equal/ReadVariableOp2F
!prune_low_magnitude_fc2_prun/cond!prune_low_magnitude_fc2_prun/cond2h
2prune_low_magnitude_fc3_prunclass/AssignVariableOp2prune_low_magnitude_fc3_prunclass/AssignVariableOp2l
4prune_low_magnitude_fc3_prunclass/AssignVariableOp_14prune_low_magnitude_fc3_prunclass/AssignVariableOp_12t
8prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp8prune_low_magnitude_fc3_prunclass/BiasAdd/ReadVariableOp2~
=prune_low_magnitude_fc3_prunclass/GreaterEqual/ReadVariableOp=prune_low_magnitude_fc3_prunclass/GreaterEqual/ReadVariableOp2x
:prune_low_magnitude_fc3_prunclass/LessEqual/ReadVariableOp:prune_low_magnitude_fc3_prunclass/LessEqual/ReadVariableOp2r
7prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp7prune_low_magnitude_fc3_prunclass/MatMul/ReadVariableOp2l
4prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp4prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp2p
6prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_16prune_low_magnitude_fc3_prunclass/Mul/ReadVariableOp_12d
0prune_low_magnitude_fc3_prunclass/ReadVariableOp0prune_low_magnitude_fc3_prunclass/ReadVariableOp2l
4prune_low_magnitude_fc3_prunclass/Sub/ReadVariableOp4prune_low_magnitude_fc3_prunclass/Sub/ReadVariableOp2�
Iprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuardIprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard2�
Eprune_low_magnitude_fc3_prunclass/assert_greater_equal/ReadVariableOpEprune_low_magnitude_fc3_prunclass/assert_greater_equal/ReadVariableOp2P
&prune_low_magnitude_fc3_prunclass/cond&prune_low_magnitude_fc3_prunclass/cond2b
/prune_low_magnitude_fc3_pruned/AssignVariableOp/prune_low_magnitude_fc3_pruned/AssignVariableOp2f
1prune_low_magnitude_fc3_pruned/AssignVariableOp_11prune_low_magnitude_fc3_pruned/AssignVariableOp_12n
5prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp5prune_low_magnitude_fc3_pruned/BiasAdd/ReadVariableOp2x
:prune_low_magnitude_fc3_pruned/GreaterEqual/ReadVariableOp:prune_low_magnitude_fc3_pruned/GreaterEqual/ReadVariableOp2r
7prune_low_magnitude_fc3_pruned/LessEqual/ReadVariableOp7prune_low_magnitude_fc3_pruned/LessEqual/ReadVariableOp2l
4prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp4prune_low_magnitude_fc3_pruned/MatMul/ReadVariableOp2f
1prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp1prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp2j
3prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_13prune_low_magnitude_fc3_pruned/Mul/ReadVariableOp_12^
-prune_low_magnitude_fc3_pruned/ReadVariableOp-prune_low_magnitude_fc3_pruned/ReadVariableOp2f
1prune_low_magnitude_fc3_pruned/Sub/ReadVariableOp1prune_low_magnitude_fc3_pruned/Sub/ReadVariableOp2�
Fprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuardFprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard2�
Bprune_low_magnitude_fc3_pruned/assert_greater_equal/ReadVariableOpBprune_low_magnitude_fc3_pruned/assert_greater_equal/ReadVariableOp2J
#prune_low_magnitude_fc3_pruned/cond#prune_low_magnitude_fc3_pruned/cond:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�E
�
C__inference_model_1_layer_call_and_return_conditional_losses_494529
encoder_input

fc1_494464:@

fc1_494466:5
#prune_low_magnitude_fc2_prun_494469:5
#prune_low_magnitude_fc2_prun_494471:1
#prune_low_magnitude_fc2_prun_494473:7
%prune_low_magnitude_fc2_1_prun_494476:7
%prune_low_magnitude_fc2_1_prun_494478:3
%prune_low_magnitude_fc2_1_prun_494480:'
encoder_output_494483:#
encoder_output_494485:7
%prune_low_magnitude_fc3_pruned_494488:7
%prune_low_magnitude_fc3_pruned_494490:3
%prune_low_magnitude_fc3_pruned_494492::
(prune_low_magnitude_fc3_prunclass_494495::
(prune_low_magnitude_fc3_prunclass_494497:6
(prune_low_magnitude_fc3_prunclass_494499:

fc4_494502: 

fc4_494504: "
fc4_class_494507:(
fc4_class_494509:(
fc4_1_494512:  
fc4_1_494514: *
classifier_output_494517:(
&
classifier_output_494519:
&
ecoder_output_494522: @"
ecoder_output_494524:@
identity

identity_1��)classifier_output/StatefulPartitionedCall�%ecoder_output/StatefulPartitionedCall�&encoder_output/StatefulPartitionedCall�fc1/StatefulPartitionedCall�fc4.1/StatefulPartitionedCall�fc4/StatefulPartitionedCall�!fc4_class/StatefulPartitionedCall�6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall�4prune_low_magnitude_fc2_prun/StatefulPartitionedCall�9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall�6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCallencoder_input
fc1_494464
fc1_494466*
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
GPU 2J 8� *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_493151�
4prune_low_magnitude_fc2_prun/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#prune_low_magnitude_fc2_prun_494469#prune_low_magnitude_fc2_prun_494471#prune_low_magnitude_fc2_prun_494473*
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
GPU 2J 8� *a
f\RZ
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_493172�
6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_fc2_prun/StatefulPartitionedCall:output:0%prune_low_magnitude_fc2_1_prun_494476%prune_low_magnitude_fc2_1_prun_494478%prune_low_magnitude_fc2_1_prun_494480*
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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_493195�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall?prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall:output:0encoder_output_494483encoder_output_494485*
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
GPU 2J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_493214�
6prune_low_magnitude_fc3_pruned/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0%prune_low_magnitude_fc3_pruned_494488%prune_low_magnitude_fc3_pruned_494490%prune_low_magnitude_fc3_pruned_494492*
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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_493235�
9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0(prune_low_magnitude_fc3_prunclass_494495(prune_low_magnitude_fc3_prunclass_494497(prune_low_magnitude_fc3_prunclass_494499*
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
GPU 2J 8� *f
faR_
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_493258�
fc4/StatefulPartitionedCallStatefulPartitionedCall?prune_low_magnitude_fc3_pruned/StatefulPartitionedCall:output:0
fc4_494502
fc4_494504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_fc4_layer_call_and_return_conditional_losses_493277�
!fc4_class/StatefulPartitionedCallStatefulPartitionedCallBprune_low_magnitude_fc3_prunclass/StatefulPartitionedCall:output:0fc4_class_494507fc4_class_494509*
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
GPU 2J 8� *N
fIRG
E__inference_fc4_class_layer_call_and_return_conditional_losses_493294�
fc4.1/StatefulPartitionedCallStatefulPartitionedCall$fc4/StatefulPartitionedCall:output:0fc4_1_494512fc4_1_494514*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc4.1_layer_call_and_return_conditional_losses_493311�
)classifier_output/StatefulPartitionedCallStatefulPartitionedCall*fc4_class/StatefulPartitionedCall:output:0classifier_output_494517classifier_output_494519*
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
GPU 2J 8� *V
fQRO
M__inference_classifier_output_layer_call_and_return_conditional_losses_493328�
%ecoder_output/StatefulPartitionedCallStatefulPartitionedCall&fc4.1/StatefulPartitionedCall:output:0ecoder_output_494522ecoder_output_494524*
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
GPU 2J 8� *R
fMRK
I__inference_ecoder_output_layer_call_and_return_conditional_losses_493345}
IdentityIdentity.ecoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�

Identity_1Identity2classifier_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp*^classifier_output/StatefulPartitionedCall&^ecoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc4.1/StatefulPartitionedCall^fc4/StatefulPartitionedCall"^fc4_class/StatefulPartitionedCall7^prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall5^prune_low_magnitude_fc2_prun/StatefulPartitionedCall:^prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall7^prune_low_magnitude_fc3_pruned/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)classifier_output/StatefulPartitionedCall)classifier_output/StatefulPartitionedCall2N
%ecoder_output/StatefulPartitionedCall%ecoder_output/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2>
fc4.1/StatefulPartitionedCallfc4.1/StatefulPartitionedCall2:
fc4/StatefulPartitionedCallfc4/StatefulPartitionedCall2F
!fc4_class/StatefulPartitionedCall!fc4_class/StatefulPartitionedCall2p
6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall2l
4prune_low_magnitude_fc2_prun/StatefulPartitionedCall4prune_low_magnitude_fc2_prun/StatefulPartitionedCall2v
9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall2p
6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�
�
Rprune_low_magnitude_fc3_pruned_assert_greater_equal_Assert_AssertGuard_true_495223�
�prune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc3_pruned_assert_greater_equal_all
V
Rprune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_placeholder	X
Tprune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_placeholder_1	U
Qprune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_identity_1
�
Kprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Oprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_fc3_pruned_assert_greater_equal_allL^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Qprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityXprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "�
Qprune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_identity_1Zprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
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
�	
�
3prune_low_magnitude_fc3_prunclass_cond_false_4954056
2prune_low_magnitude_fc3_prunclass_cond_placeholder8
4prune_low_magnitude_fc3_prunclass_cond_placeholder_18
4prune_low_magnitude_fc3_prunclass_cond_placeholder_28
4prune_low_magnitude_fc3_prunclass_cond_placeholder_3b
^prune_low_magnitude_fc3_prunclass_cond_identity_prune_low_magnitude_fc3_prunclass_logicaland_1
5
1prune_low_magnitude_fc3_prunclass_cond_identity_1
q
+prune_low_magnitude_fc3_prunclass/cond/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
/prune_low_magnitude_fc3_prunclass/cond/IdentityIdentity^prune_low_magnitude_fc3_prunclass_cond_identity_prune_low_magnitude_fc3_prunclass_logicaland_1,^prune_low_magnitude_fc3_prunclass/cond/NoOp*
T0
*
_output_shapes
: �
1prune_low_magnitude_fc3_prunclass/cond/Identity_1Identity8prune_low_magnitude_fc3_prunclass/cond/Identity:output:0*
T0
*
_output_shapes
: "o
1prune_low_magnitude_fc3_prunclass_cond_identity_1:prune_low_magnitude_fc3_prunclass/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
�F
�
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_494145

inputs!
readvariableop_resource:	 
cond_input_1:
cond_input_2:
cond_input_3: -
biasadd_readvariableop_resource:
identity��AssignVariableOp�AssignVariableOp_1�BiasAdd/ReadVariableOp�GreaterEqual/ReadVariableOp�LessEqual/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1�ReadVariableOp�Sub/ReadVariableOp�'assert_greater_equal/Assert/AssertGuard�#assert_greater_equal/ReadVariableOp�cond^
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
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_494020*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_494019�
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
 *��Y?�
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
*$
else_branchR
cond_false_494060*
output_shapes
: *#
then_branchR
cond_true_494059q
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
 w
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp_1*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_120
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12 
ReadVariableOpReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4assert_greater_equal_Assert_AssertGuard_false_496205K
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
cond_false_495647
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
��
�<
"__inference__traced_restore_496984
file_prefix-
assignvariableop_fc1_kernel:@)
assignvariableop_1_fc1_bias:F
4assignvariableop_2_prune_low_magnitude_fc2_prun_mask:C
9assignvariableop_3_prune_low_magnitude_fc2_prun_threshold: F
<assignvariableop_4_prune_low_magnitude_fc2_prun_pruning_step:	 H
6assignvariableop_5_prune_low_magnitude_fc2_1_prun_mask:E
;assignvariableop_6_prune_low_magnitude_fc2_1_prun_threshold: H
>assignvariableop_7_prune_low_magnitude_fc2_1_prun_pruning_step:	 :
(assignvariableop_8_encoder_output_kernel:4
&assignvariableop_9_encoder_output_bias:I
7assignvariableop_10_prune_low_magnitude_fc3_pruned_mask:F
<assignvariableop_11_prune_low_magnitude_fc3_pruned_threshold: I
?assignvariableop_12_prune_low_magnitude_fc3_pruned_pruning_step:	 0
assignvariableop_13_fc4_kernel: *
assignvariableop_14_fc4_bias: L
:assignvariableop_15_prune_low_magnitude_fc3_prunclass_mask:I
?assignvariableop_16_prune_low_magnitude_fc3_prunclass_threshold: L
Bassignvariableop_17_prune_low_magnitude_fc3_prunclass_pruning_step:	 2
 assignvariableop_18_fc4_1_kernel:  ,
assignvariableop_19_fc4_1_bias: 6
$assignvariableop_20_fc4_class_kernel:(0
"assignvariableop_21_fc4_class_bias:(:
(assignvariableop_22_ecoder_output_kernel: @4
&assignvariableop_23_ecoder_output_bias:@>
,assignvariableop_24_classifier_output_kernel:(
8
*assignvariableop_25_classifier_output_bias:
I
7assignvariableop_26_prune_low_magnitude_fc2_prun_kernel:C
5assignvariableop_27_prune_low_magnitude_fc2_prun_bias:K
9assignvariableop_28_prune_low_magnitude_fc2_1_prun_kernel:E
7assignvariableop_29_prune_low_magnitude_fc2_1_prun_bias:K
9assignvariableop_30_prune_low_magnitude_fc3_pruned_kernel:E
7assignvariableop_31_prune_low_magnitude_fc3_pruned_bias:N
<assignvariableop_32_prune_low_magnitude_fc3_prunclass_kernel:H
:assignvariableop_33_prune_low_magnitude_fc3_prunclass_bias:'
assignvariableop_34_iteration:	 +
!assignvariableop_35_learning_rate: 7
%assignvariableop_36_adam_m_fc1_kernel:@7
%assignvariableop_37_adam_v_fc1_kernel:@1
#assignvariableop_38_adam_m_fc1_bias:1
#assignvariableop_39_adam_v_fc1_bias:P
>assignvariableop_40_adam_m_prune_low_magnitude_fc2_prun_kernel:P
>assignvariableop_41_adam_v_prune_low_magnitude_fc2_prun_kernel:J
<assignvariableop_42_adam_m_prune_low_magnitude_fc2_prun_bias:J
<assignvariableop_43_adam_v_prune_low_magnitude_fc2_prun_bias:R
@assignvariableop_44_adam_m_prune_low_magnitude_fc2_1_prun_kernel:R
@assignvariableop_45_adam_v_prune_low_magnitude_fc2_1_prun_kernel:L
>assignvariableop_46_adam_m_prune_low_magnitude_fc2_1_prun_bias:L
>assignvariableop_47_adam_v_prune_low_magnitude_fc2_1_prun_bias:B
0assignvariableop_48_adam_m_encoder_output_kernel:B
0assignvariableop_49_adam_v_encoder_output_kernel:<
.assignvariableop_50_adam_m_encoder_output_bias:<
.assignvariableop_51_adam_v_encoder_output_bias:R
@assignvariableop_52_adam_m_prune_low_magnitude_fc3_pruned_kernel:R
@assignvariableop_53_adam_v_prune_low_magnitude_fc3_pruned_kernel:L
>assignvariableop_54_adam_m_prune_low_magnitude_fc3_pruned_bias:L
>assignvariableop_55_adam_v_prune_low_magnitude_fc3_pruned_bias:7
%assignvariableop_56_adam_m_fc4_kernel: 7
%assignvariableop_57_adam_v_fc4_kernel: 1
#assignvariableop_58_adam_m_fc4_bias: 1
#assignvariableop_59_adam_v_fc4_bias: U
Cassignvariableop_60_adam_m_prune_low_magnitude_fc3_prunclass_kernel:U
Cassignvariableop_61_adam_v_prune_low_magnitude_fc3_prunclass_kernel:O
Aassignvariableop_62_adam_m_prune_low_magnitude_fc3_prunclass_bias:O
Aassignvariableop_63_adam_v_prune_low_magnitude_fc3_prunclass_bias:9
'assignvariableop_64_adam_m_fc4_1_kernel:  9
'assignvariableop_65_adam_v_fc4_1_kernel:  3
%assignvariableop_66_adam_m_fc4_1_bias: 3
%assignvariableop_67_adam_v_fc4_1_bias: =
+assignvariableop_68_adam_m_fc4_class_kernel:(=
+assignvariableop_69_adam_v_fc4_class_kernel:(7
)assignvariableop_70_adam_m_fc4_class_bias:(7
)assignvariableop_71_adam_v_fc4_class_bias:(A
/assignvariableop_72_adam_m_ecoder_output_kernel: @A
/assignvariableop_73_adam_v_ecoder_output_kernel: @;
-assignvariableop_74_adam_m_ecoder_output_bias:@;
-assignvariableop_75_adam_v_ecoder_output_bias:@E
3assignvariableop_76_adam_m_classifier_output_kernel:(
E
3assignvariableop_77_adam_v_classifier_output_kernel:(
?
1assignvariableop_78_adam_m_classifier_output_bias:
?
1assignvariableop_79_adam_v_classifier_output_bias:
%
assignvariableop_80_total_4: %
assignvariableop_81_count_4: %
assignvariableop_82_total_3: %
assignvariableop_83_count_3: %
assignvariableop_84_total_2: %
assignvariableop_85_count_2: %
assignvariableop_86_total_1: %
assignvariableop_87_count_1: #
assignvariableop_88_total: #
assignvariableop_89_count: 
identity_91��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*�%
value�%B�%[B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-2/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-4/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-6/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-6/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*�
value�B�[B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*i
dtypes_
]2[					[
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
AssignVariableOp_5AssignVariableOp6assignvariableop_5_prune_low_magnitude_fc2_1_prun_maskIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp;assignvariableop_6_prune_low_magnitude_fc2_1_prun_thresholdIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp>assignvariableop_7_prune_low_magnitude_fc2_1_prun_pruning_stepIdentity_7:output:0"/device:CPU:0*&
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
AssignVariableOp_10AssignVariableOp7assignvariableop_10_prune_low_magnitude_fc3_pruned_maskIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp<assignvariableop_11_prune_low_magnitude_fc3_pruned_thresholdIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp?assignvariableop_12_prune_low_magnitude_fc3_pruned_pruning_stepIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_fc4_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_fc4_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp:assignvariableop_15_prune_low_magnitude_fc3_prunclass_maskIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp?assignvariableop_16_prune_low_magnitude_fc3_prunclass_thresholdIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpBassignvariableop_17_prune_low_magnitude_fc3_prunclass_pruning_stepIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp assignvariableop_18_fc4_1_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_fc4_1_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp$assignvariableop_20_fc4_class_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_fc4_class_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_ecoder_output_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp&assignvariableop_23_ecoder_output_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_classifier_output_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_classifier_output_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp7assignvariableop_26_prune_low_magnitude_fc2_prun_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp5assignvariableop_27_prune_low_magnitude_fc2_prun_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp9assignvariableop_28_prune_low_magnitude_fc2_1_prun_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp7assignvariableop_29_prune_low_magnitude_fc2_1_prun_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp9assignvariableop_30_prune_low_magnitude_fc3_pruned_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp7assignvariableop_31_prune_low_magnitude_fc3_pruned_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp<assignvariableop_32_prune_low_magnitude_fc3_prunclass_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp:assignvariableop_33_prune_low_magnitude_fc3_prunclass_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_iterationIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp!assignvariableop_35_learning_rateIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_m_fc1_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_v_fc1_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp#assignvariableop_38_adam_m_fc1_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp#assignvariableop_39_adam_v_fc1_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_m_prune_low_magnitude_fc2_prun_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp>assignvariableop_41_adam_v_prune_low_magnitude_fc2_prun_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp<assignvariableop_42_adam_m_prune_low_magnitude_fc2_prun_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp<assignvariableop_43_adam_v_prune_low_magnitude_fc2_prun_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp@assignvariableop_44_adam_m_prune_low_magnitude_fc2_1_prun_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp@assignvariableop_45_adam_v_prune_low_magnitude_fc2_1_prun_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp>assignvariableop_46_adam_m_prune_low_magnitude_fc2_1_prun_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp>assignvariableop_47_adam_v_prune_low_magnitude_fc2_1_prun_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp0assignvariableop_48_adam_m_encoder_output_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp0assignvariableop_49_adam_v_encoder_output_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp.assignvariableop_50_adam_m_encoder_output_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp.assignvariableop_51_adam_v_encoder_output_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp@assignvariableop_52_adam_m_prune_low_magnitude_fc3_pruned_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp@assignvariableop_53_adam_v_prune_low_magnitude_fc3_pruned_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp>assignvariableop_54_adam_m_prune_low_magnitude_fc3_pruned_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp>assignvariableop_55_adam_v_prune_low_magnitude_fc3_pruned_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp%assignvariableop_56_adam_m_fc4_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp%assignvariableop_57_adam_v_fc4_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp#assignvariableop_58_adam_m_fc4_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp#assignvariableop_59_adam_v_fc4_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpCassignvariableop_60_adam_m_prune_low_magnitude_fc3_prunclass_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpCassignvariableop_61_adam_v_prune_low_magnitude_fc3_prunclass_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpAassignvariableop_62_adam_m_prune_low_magnitude_fc3_prunclass_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpAassignvariableop_63_adam_v_prune_low_magnitude_fc3_prunclass_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_m_fc4_1_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_v_fc4_1_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp%assignvariableop_66_adam_m_fc4_1_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp%assignvariableop_67_adam_v_fc4_1_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_m_fc4_class_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_v_fc4_class_kernelIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_m_fc4_class_biasIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_v_fc4_class_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp/assignvariableop_72_adam_m_ecoder_output_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp/assignvariableop_73_adam_v_ecoder_output_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp-assignvariableop_74_adam_m_ecoder_output_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp-assignvariableop_75_adam_v_ecoder_output_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp3assignvariableop_76_adam_m_classifier_output_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp3assignvariableop_77_adam_v_classifier_output_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp1assignvariableop_78_adam_m_classifier_output_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp1assignvariableop_79_adam_v_classifier_output_biasIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOpassignvariableop_80_total_4Identity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOpassignvariableop_81_count_4Identity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOpassignvariableop_82_total_3Identity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOpassignvariableop_83_count_3Identity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOpassignvariableop_84_total_2Identity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOpassignvariableop_85_count_2Identity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpassignvariableop_86_total_1Identity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOpassignvariableop_87_count_1Identity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOpassignvariableop_88_totalIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOpassignvariableop_89_countIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_90Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_91IdentityIdentity_90:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_91Identity_91:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�>
�
cond_true_4958323
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
 *��Y?�
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
�>
�
cond_true_4962443
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
 *��Y?�
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
�
�
cond_false_494060
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
�
�
4assert_greater_equal_Assert_AssertGuard_false_494020K
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
M__inference_classifier_output_layer_call_and_return_conditional_losses_496410

inputs0
matmul_readvariableop_resource:(
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�c
�
-prune_low_magnitude_fc2_prun_cond_true_494974P
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
 *��Y?�
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
3assert_greater_equal_Assert_AssertGuard_true_493847M
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
�
=__inference_prune_low_magnitude_fc2_prun_layer_call_fn_495572

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
GPU 2J 8� *a
f\RZ
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_494145o
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
�
�
Vprune_low_magnitude_fc3_prunclass_assert_greater_equal_Assert_AssertGuard_false_495365�
�prune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prunclass_assert_greater_equal_all
�
�prune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prunclass_assert_greater_equal_readvariableop	�
�prune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prunclass_assert_greater_equal_y	X
Tprune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_identity_1
��Pprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert�
Wprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Wprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Wprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*_
valueVBT BNx (prune_low_magnitude_fc3_prunclass/assert_greater_equal/ReadVariableOp:0) = �
Wprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*R
valueIBG BAy (prune_low_magnitude_fc3_prunclass/assert_greater_equal/y:0) = �
Pprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/AssertAssert�prune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prunclass_assert_greater_equal_all`prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0`prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0`prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0�prune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prunclass_assert_greater_equal_readvariableop`prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0�prune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prunclass_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Rprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_prunclass_assert_greater_equal_allQ^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
Tprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity_1Identity[prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity:output:0O^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Nprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/NoOpNoOpQ^prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "�
Tprune_low_magnitude_fc3_prunclass_assert_greater_equal_assert_assertguard_identity_1]prune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2�
Pprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/AssertPprune_low_magnitude_fc3_prunclass/assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_496185

inputs-
mul_readvariableop_resource:/
mul_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��AssignVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Mul/ReadVariableOp�Mul/ReadVariableOp_1'
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
 �
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_classifier_output_layer_call_fn_496399

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
GPU 2J 8� *V
fQRO
M__inference_classifier_output_layer_call_and_return_conditional_losses_493328o
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
�
�
Sprune_low_magnitude_fc3_pruned_assert_greater_equal_Assert_AssertGuard_false_495224�
�prune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_pruned_assert_greater_equal_all
�
�prune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_pruned_assert_greater_equal_readvariableop	�
�prune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_pruned_assert_greater_equal_y	U
Qprune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_identity_1
��Mprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert�
Tprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.�
Tprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:�
Tprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*\
valueSBQ BKx (prune_low_magnitude_fc3_pruned/assert_greater_equal/ReadVariableOp:0) = �
Tprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (prune_low_magnitude_fc3_pruned/assert_greater_equal/y:0) = �
Mprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/AssertAssert�prune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_pruned_assert_greater_equal_all]prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0]prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0]prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0�prune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_pruned_assert_greater_equal_readvariableop]prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0�prune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_pruned_assert_greater_equal_y*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Oprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/IdentityIdentity�prune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_fc3_pruned_assert_greater_equal_allN^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: �
Qprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityXprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity:output:0L^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: �
Kprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/NoOpNoOpN^prune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "�
Qprune_low_magnitude_fc3_pruned_assert_greater_equal_assert_assertguard_identity_1Zprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2�
Mprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/AssertMprune_low_magnitude_fc3_pruned/assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_model_1_layer_call_fn_493410
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

unknown_12:

unknown_13:

unknown_14:

unknown_15: 

unknown_16: 

unknown_17:(

unknown_18:(

unknown_19:  

unknown_20: 

unknown_21:(


unknown_22:


unknown_23: @

unknown_24:@
identity

identity_1��StatefulPartitionedCall�
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
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������
*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_493353o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_nameencoder_input
�E
�
C__inference_model_1_layer_call_and_return_conditional_losses_493353

inputs

fc1_493152:@

fc1_493154:5
#prune_low_magnitude_fc2_prun_493173:5
#prune_low_magnitude_fc2_prun_493175:1
#prune_low_magnitude_fc2_prun_493177:7
%prune_low_magnitude_fc2_1_prun_493196:7
%prune_low_magnitude_fc2_1_prun_493198:3
%prune_low_magnitude_fc2_1_prun_493200:'
encoder_output_493215:#
encoder_output_493217:7
%prune_low_magnitude_fc3_pruned_493236:7
%prune_low_magnitude_fc3_pruned_493238:3
%prune_low_magnitude_fc3_pruned_493240::
(prune_low_magnitude_fc3_prunclass_493259::
(prune_low_magnitude_fc3_prunclass_493261:6
(prune_low_magnitude_fc3_prunclass_493263:

fc4_493278: 

fc4_493280: "
fc4_class_493295:(
fc4_class_493297:(
fc4_1_493312:  
fc4_1_493314: *
classifier_output_493329:(
&
classifier_output_493331:
&
ecoder_output_493346: @"
ecoder_output_493348:@
identity

identity_1��)classifier_output/StatefulPartitionedCall�%ecoder_output/StatefulPartitionedCall�&encoder_output/StatefulPartitionedCall�fc1/StatefulPartitionedCall�fc4.1/StatefulPartitionedCall�fc4/StatefulPartitionedCall�!fc4_class/StatefulPartitionedCall�6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall�4prune_low_magnitude_fc2_prun/StatefulPartitionedCall�9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall�6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCallinputs
fc1_493152
fc1_493154*
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
GPU 2J 8� *H
fCRA
?__inference_fc1_layer_call_and_return_conditional_losses_493151�
4prune_low_magnitude_fc2_prun/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#prune_low_magnitude_fc2_prun_493173#prune_low_magnitude_fc2_prun_493175#prune_low_magnitude_fc2_prun_493177*
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
GPU 2J 8� *a
f\RZ
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_493172�
6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCallStatefulPartitionedCall=prune_low_magnitude_fc2_prun/StatefulPartitionedCall:output:0%prune_low_magnitude_fc2_1_prun_493196%prune_low_magnitude_fc2_1_prun_493198%prune_low_magnitude_fc2_1_prun_493200*
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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_493195�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall?prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall:output:0encoder_output_493215encoder_output_493217*
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
GPU 2J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_493214�
6prune_low_magnitude_fc3_pruned/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0%prune_low_magnitude_fc3_pruned_493236%prune_low_magnitude_fc3_pruned_493238%prune_low_magnitude_fc3_pruned_493240*
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
GPU 2J 8� *c
f^R\
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_493235�
9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0(prune_low_magnitude_fc3_prunclass_493259(prune_low_magnitude_fc3_prunclass_493261(prune_low_magnitude_fc3_prunclass_493263*
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
GPU 2J 8� *f
faR_
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_493258�
fc4/StatefulPartitionedCallStatefulPartitionedCall?prune_low_magnitude_fc3_pruned/StatefulPartitionedCall:output:0
fc4_493278
fc4_493280*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_fc4_layer_call_and_return_conditional_losses_493277�
!fc4_class/StatefulPartitionedCallStatefulPartitionedCallBprune_low_magnitude_fc3_prunclass/StatefulPartitionedCall:output:0fc4_class_493295fc4_class_493297*
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
GPU 2J 8� *N
fIRG
E__inference_fc4_class_layer_call_and_return_conditional_losses_493294�
fc4.1/StatefulPartitionedCallStatefulPartitionedCall$fc4/StatefulPartitionedCall:output:0fc4_1_493312fc4_1_493314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc4.1_layer_call_and_return_conditional_losses_493311�
)classifier_output/StatefulPartitionedCallStatefulPartitionedCall*fc4_class/StatefulPartitionedCall:output:0classifier_output_493329classifier_output_493331*
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
GPU 2J 8� *V
fQRO
M__inference_classifier_output_layer_call_and_return_conditional_losses_493328�
%ecoder_output/StatefulPartitionedCallStatefulPartitionedCall&fc4.1/StatefulPartitionedCall:output:0ecoder_output_493346ecoder_output_493348*
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
GPU 2J 8� *R
fMRK
I__inference_ecoder_output_layer_call_and_return_conditional_losses_493345}
IdentityIdentity.ecoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�

Identity_1Identity2classifier_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp*^classifier_output/StatefulPartitionedCall&^ecoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc4.1/StatefulPartitionedCall^fc4/StatefulPartitionedCall"^fc4_class/StatefulPartitionedCall7^prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall5^prune_low_magnitude_fc2_prun/StatefulPartitionedCall:^prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall7^prune_low_magnitude_fc3_pruned/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)classifier_output/StatefulPartitionedCall)classifier_output/StatefulPartitionedCall2N
%ecoder_output/StatefulPartitionedCall%ecoder_output/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2>
fc4.1/StatefulPartitionedCallfc4.1/StatefulPartitionedCall2:
fc4/StatefulPartitionedCallfc4/StatefulPartitionedCall2F
!fc4_class/StatefulPartitionedCall!fc4_class/StatefulPartitionedCall2p
6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall6prune_low_magnitude_fc2.1_prun/StatefulPartitionedCall2l
4prune_low_magnitude_fc2_prun/StatefulPartitionedCall4prune_low_magnitude_fc2_prun/StatefulPartitionedCall2v
9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall9prune_low_magnitude_fc3_prunclass/StatefulPartitionedCall2p
6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall6prune_low_magnitude_fc3_pruned/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
4assert_greater_equal_Assert_AssertGuard_false_495607K
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
: "�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
encoder_input6
serving_default_encoder_input:0���������@E
classifier_output0
StatefulPartitionedCall:0���������
A
ecoder_output0
StatefulPartitionedCall:1���������@tensorflow/serving/predict:��
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%pruning_vars
	&layer
'prunable_weights
(mask
)	threshold
*pruning_step"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1pruning_vars
	2layer
3prunable_weights
4mask
5	threshold
6pruning_step"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Epruning_vars
	Flayer
Gprunable_weights
Hmask
I	threshold
Jpruning_step"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Ypruning_vars
	Zlayer
[prunable_weights
\mask
]	threshold
^pruning_step"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias"
_tf_keras_layer
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias"
_tf_keras_layer
�
0
1
2
�3
(4
)5
*6
�7
�8
49
510
611
=12
>13
�14
�15
H16
I17
J18
Q19
R20
�21
�22
\23
]24
^25
e26
f27
m28
n29
u30
v31
}32
~33"
trackable_list_wrapper
�
0
1
2
�3
�4
�5
=6
>7
�8
�9
Q10
R11
�12
�13
e14
f15
m16
n17
u18
v19
}20
~21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
(__inference_model_1_layer_call_fn_493410
(__inference_model_1_layer_call_fn_494735
(__inference_model_1_layer_call_fn_494810
(__inference_model_1_layer_call_fn_494461�
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
C__inference_model_1_layer_call_and_return_conditional_losses_494908
C__inference_model_1_layer_call_and_return_conditional_losses_495526
C__inference_model_1_layer_call_and_return_conditional_losses_494529
C__inference_model_1_layer_call_and_return_conditional_losses_494613�
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
!__inference__wrapped_model_493133encoder_input"�
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
 "
trackable_list_wrapper
-
�serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_fc1_layer_call_fn_495535�
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
?__inference_fc1_layer_call_and_return_conditional_losses_495546�
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
:@2
fc1/kernel
:2fc1/bias
D
0
�1
(2
)3
*4"
trackable_list_wrapper
/
0
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
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
=__inference_prune_low_magnitude_fc2_prun_layer_call_fn_495557
=__inference_prune_low_magnitude_fc2_prun_layer_call_fn_495572�
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
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_495587
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_495732�
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
	�bias"
_tf_keras_layer
'
0"
trackable_list_wrapper
3:1(2!prune_low_magnitude_fc2_prun/mask
0:. (2&prune_low_magnitude_fc2_prun/threshold
1:/	 2)prune_low_magnitude_fc2_prun/pruning_step
E
�0
�1
42
53
64"
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
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
?__inference_prune_low_magnitude_fc2.1_prun_layer_call_fn_495743
?__inference_prune_low_magnitude_fc2.1_prun_layer_call_fn_495758�
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
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_495773
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_495918�
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
(
�0"
trackable_list_wrapper
5:3(2#prune_low_magnitude_fc2.1_prun/mask
2:0 (2(prune_low_magnitude_fc2.1_prun/threshold
3:1	 2+prune_low_magnitude_fc2.1_prun/pruning_step
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_encoder_output_layer_call_fn_495927�
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
J__inference_encoder_output_layer_call_and_return_conditional_losses_495938�
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
H2
I3
J4"
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
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
?__inference_prune_low_magnitude_fc3_pruned_layer_call_fn_495949
?__inference_prune_low_magnitude_fc3_pruned_layer_call_fn_495964�
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
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_495979
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_496124�
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
(
�0"
trackable_list_wrapper
5:3(2#prune_low_magnitude_fc3_pruned/mask
2:0 (2(prune_low_magnitude_fc3_pruned/threshold
3:1	 2+prune_low_magnitude_fc3_pruned/pruning_step
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_fc4_layer_call_fn_496133�
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
?__inference_fc4_layer_call_and_return_conditional_losses_496144�
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
: 2
fc4/kernel
: 2fc4/bias
E
�0
�1
\2
]3
^4"
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
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
B__inference_prune_low_magnitude_fc3_prunclass_layer_call_fn_496155
B__inference_prune_low_magnitude_fc3_prunclass_layer_call_fn_496170�
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
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_496185
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_496330�
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
(
�0"
trackable_list_wrapper
8:6(2&prune_low_magnitude_fc3_prunclass/mask
5:3 (2+prune_low_magnitude_fc3_prunclass/threshold
6:4	 2.prune_low_magnitude_fc3_prunclass/pruning_step
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_fc4.1_layer_call_fn_496339�
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
A__inference_fc4.1_layer_call_and_return_conditional_losses_496350�
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
:  2fc4.1/kernel
: 2
fc4.1/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_fc4_class_layer_call_fn_496359�
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
E__inference_fc4_class_layer_call_and_return_conditional_losses_496370�
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
": (2fc4_class/kernel
:(2fc4_class/bias
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_ecoder_output_layer_call_fn_496379�
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
I__inference_ecoder_output_layer_call_and_return_conditional_losses_496390�
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
&:$ @2ecoder_output/kernel
 :@2ecoder_output/bias
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_classifier_output_layer_call_fn_496399�
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
M__inference_classifier_output_layer_call_and_return_conditional_losses_496410�
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
*:((
2classifier_output/kernel
$:"
2classifier_output/bias
5:32#prune_low_magnitude_fc2_prun/kernel
/:-2!prune_low_magnitude_fc2_prun/bias
7:52%prune_low_magnitude_fc2.1_prun/kernel
1:/2#prune_low_magnitude_fc2.1_prun/bias
7:52%prune_low_magnitude_fc3_pruned/kernel
1:/2#prune_low_magnitude_fc3_pruned/bias
::82(prune_low_magnitude_fc3_prunclass/kernel
4:22&prune_low_magnitude_fc3_prunclass/bias
v
(0
)1
*2
43
54
65
H6
I7
J8
\9
]10
^11"
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_1_layer_call_fn_493410encoder_input"�
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
(__inference_model_1_layer_call_fn_494735inputs"�
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
(__inference_model_1_layer_call_fn_494810inputs"�
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
�B�
(__inference_model_1_layer_call_fn_494461encoder_input"�
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
C__inference_model_1_layer_call_and_return_conditional_losses_494908inputs"�
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
C__inference_model_1_layer_call_and_return_conditional_losses_495526inputs"�
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
C__inference_model_1_layer_call_and_return_conditional_losses_494529encoder_input"�
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
C__inference_model_1_layer_call_and_return_conditional_losses_494613encoder_input"�
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
�
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
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44"
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
�13
�14
�15
�16
�17
�18
�19
�20
�21"
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
�13
�14
�15
�16
�17
�18
�19
�20
�21"
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
$__inference_signature_wrapper_494676encoder_input"�
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
$__inference_fc1_layer_call_fn_495535inputs"�
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
?__inference_fc1_layer_call_and_return_conditional_losses_495546inputs"�
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
(0
)1
*2"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
=__inference_prune_low_magnitude_fc2_prun_layer_call_fn_495557inputs"�
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
=__inference_prune_low_magnitude_fc2_prun_layer_call_fn_495572inputs"�
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
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_495587inputs"�
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
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_495732inputs"�
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
6
0
(1
)2"
trackable_tuple_wrapper
/
0
�1"
trackable_list_wrapper
/
0
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
5
40
51
62"
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
?__inference_prune_low_magnitude_fc2.1_prun_layer_call_fn_495743inputs"�
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
?__inference_prune_low_magnitude_fc2.1_prun_layer_call_fn_495758inputs"�
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
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_495773inputs"�
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
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_495918inputs"�
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
41
52"
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
/__inference_encoder_output_layer_call_fn_495927inputs"�
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
J__inference_encoder_output_layer_call_and_return_conditional_losses_495938inputs"�
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
H0
I1
J2"
trackable_list_wrapper
'
F0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
?__inference_prune_low_magnitude_fc3_pruned_layer_call_fn_495949inputs"�
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
?__inference_prune_low_magnitude_fc3_pruned_layer_call_fn_495964inputs"�
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
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_495979inputs"�
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
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_496124inputs"�
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
H1
I2"
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
$__inference_fc4_layer_call_fn_496133inputs"�
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
?__inference_fc4_layer_call_and_return_conditional_losses_496144inputs"�
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
\0
]1
^2"
trackable_list_wrapper
'
Z0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
B__inference_prune_low_magnitude_fc3_prunclass_layer_call_fn_496155inputs"�
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
B__inference_prune_low_magnitude_fc3_prunclass_layer_call_fn_496170inputs"�
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
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_496185inputs"�
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
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_496330inputs"�
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
\1
]2"
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
&__inference_fc4.1_layer_call_fn_496339inputs"�
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
A__inference_fc4.1_layer_call_and_return_conditional_losses_496350inputs"�
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
*__inference_fc4_class_layer_call_fn_496359inputs"�
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
E__inference_fc4_class_layer_call_and_return_conditional_losses_496370inputs"�
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
.__inference_ecoder_output_layer_call_fn_496379inputs"�
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
I__inference_ecoder_output_layer_call_and_return_conditional_losses_496390inputs"�
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
2__inference_classifier_output_layer_call_fn_496399inputs"�
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
M__inference_classifier_output_layer_call_and_return_conditional_losses_496410inputs"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
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
<::2,Adam/m/prune_low_magnitude_fc2.1_prun/kernel
<::2,Adam/v/prune_low_magnitude_fc2.1_prun/kernel
6:42*Adam/m/prune_low_magnitude_fc2.1_prun/bias
6:42*Adam/v/prune_low_magnitude_fc2.1_prun/bias
,:*2Adam/m/encoder_output/kernel
,:*2Adam/v/encoder_output/kernel
&:$2Adam/m/encoder_output/bias
&:$2Adam/v/encoder_output/bias
<::2,Adam/m/prune_low_magnitude_fc3_pruned/kernel
<::2,Adam/v/prune_low_magnitude_fc3_pruned/kernel
6:42*Adam/m/prune_low_magnitude_fc3_pruned/bias
6:42*Adam/v/prune_low_magnitude_fc3_pruned/bias
!: 2Adam/m/fc4/kernel
!: 2Adam/v/fc4/kernel
: 2Adam/m/fc4/bias
: 2Adam/v/fc4/bias
?:=2/Adam/m/prune_low_magnitude_fc3_prunclass/kernel
?:=2/Adam/v/prune_low_magnitude_fc3_prunclass/kernel
9:72-Adam/m/prune_low_magnitude_fc3_prunclass/bias
9:72-Adam/v/prune_low_magnitude_fc3_prunclass/bias
#:!  2Adam/m/fc4.1/kernel
#:!  2Adam/v/fc4.1/kernel
: 2Adam/m/fc4.1/bias
: 2Adam/v/fc4.1/bias
':%(2Adam/m/fc4_class/kernel
':%(2Adam/v/fc4_class/kernel
!:(2Adam/m/fc4_class/bias
!:(2Adam/v/fc4_class/bias
+:) @2Adam/m/ecoder_output/kernel
+:) @2Adam/v/ecoder_output/kernel
%:#@2Adam/m/ecoder_output/bias
%:#@2Adam/v/ecoder_output/bias
/:-(
2Adam/m/classifier_output/kernel
/:-(
2Adam/v/classifier_output/kernel
):'
2Adam/m/classifier_output/bias
):'
2Adam/v/classifier_output/bias
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
 "
trackable_dict_wrapper�
!__inference__wrapped_model_493133�!(��4�=>�H��\�QRmnef}~uv6�3
,�)
'�$
encoder_input���������@
� "�|
@
classifier_output+�(
classifier_output���������

8
ecoder_output'�$
ecoder_output���������@�
M__inference_classifier_output_layer_call_and_return_conditional_losses_496410c}~/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������

� �
2__inference_classifier_output_layer_call_fn_496399X}~/�,
%�"
 �
inputs���������(
� "!�
unknown���������
�
I__inference_ecoder_output_layer_call_and_return_conditional_losses_496390cuv/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������@
� �
.__inference_ecoder_output_layer_call_fn_496379Xuv/�,
%�"
 �
inputs��������� 
� "!�
unknown���������@�
J__inference_encoder_output_layer_call_and_return_conditional_losses_495938c=>/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
/__inference_encoder_output_layer_call_fn_495927X=>/�,
%�"
 �
inputs���������
� "!�
unknown����������
?__inference_fc1_layer_call_and_return_conditional_losses_495546c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
$__inference_fc1_layer_call_fn_495535X/�,
%�"
 �
inputs���������@
� "!�
unknown����������
A__inference_fc4.1_layer_call_and_return_conditional_losses_496350cef/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
&__inference_fc4.1_layer_call_fn_496339Xef/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
E__inference_fc4_class_layer_call_and_return_conditional_losses_496370cmn/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������(
� �
*__inference_fc4_class_layer_call_fn_496359Xmn/�,
%�"
 �
inputs���������
� "!�
unknown���������(�
?__inference_fc4_layer_call_and_return_conditional_losses_496144cQR/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0��������� 
� �
$__inference_fc4_layer_call_fn_496133XQR/�,
%�"
 �
inputs���������
� "!�
unknown��������� �
C__inference_model_1_layer_call_and_return_conditional_losses_494529�!(��4�=>�H��\�QRmnef}~uv>�;
4�1
'�$
encoder_input���������@
p 

 
� "Y�V
O�L
$�!

tensor_0_0���������@
$�!

tensor_0_1���������

� �
C__inference_model_1_layer_call_and_return_conditional_losses_494613�)*()�6�45�=>J�HI�^�\]�QRmnef}~uv>�;
4�1
'�$
encoder_input���������@
p

 
� "Y�V
O�L
$�!

tensor_0_0���������@
$�!

tensor_0_1���������

� �
C__inference_model_1_layer_call_and_return_conditional_losses_494908�!(��4�=>�H��\�QRmnef}~uv7�4
-�*
 �
inputs���������@
p 

 
� "Y�V
O�L
$�!

tensor_0_0���������@
$�!

tensor_0_1���������

� �
C__inference_model_1_layer_call_and_return_conditional_losses_495526�)*()�6�45�=>J�HI�^�\]�QRmnef}~uv7�4
-�*
 �
inputs���������@
p

 
� "Y�V
O�L
$�!

tensor_0_0���������@
$�!

tensor_0_1���������

� �
(__inference_model_1_layer_call_fn_493410�!(��4�=>�H��\�QRmnef}~uv>�;
4�1
'�$
encoder_input���������@
p 

 
� "K�H
"�
tensor_0���������@
"�
tensor_1���������
�
(__inference_model_1_layer_call_fn_494461�)*()�6�45�=>J�HI�^�\]�QRmnef}~uv>�;
4�1
'�$
encoder_input���������@
p

 
� "K�H
"�
tensor_0���������@
"�
tensor_1���������
�
(__inference_model_1_layer_call_fn_494735�!(��4�=>�H��\�QRmnef}~uv7�4
-�*
 �
inputs���������@
p 

 
� "K�H
"�
tensor_0���������@
"�
tensor_1���������
�
(__inference_model_1_layer_call_fn_494810�)*()�6�45�=>J�HI�^�\]�QRmnef}~uv7�4
-�*
 �
inputs���������@
p

 
� "K�H
"�
tensor_0���������@
"�
tensor_1���������
�
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_495773j�4�3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Z__inference_prune_low_magnitude_fc2.1_prun_layer_call_and_return_conditional_losses_495918l6�45�3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
?__inference_prune_low_magnitude_fc2.1_prun_layer_call_fn_495743_�4�3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
?__inference_prune_low_magnitude_fc2.1_prun_layer_call_fn_495758a6�45�3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_495587i(�3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
X__inference_prune_low_magnitude_fc2_prun_layer_call_and_return_conditional_losses_495732k*()�3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
=__inference_prune_low_magnitude_fc2_prun_layer_call_fn_495557^(�3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
=__inference_prune_low_magnitude_fc2_prun_layer_call_fn_495572`*()�3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_496185j�\�3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
]__inference_prune_low_magnitude_fc3_prunclass_layer_call_and_return_conditional_losses_496330l^�\]�3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
B__inference_prune_low_magnitude_fc3_prunclass_layer_call_fn_496155_�\�3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
B__inference_prune_low_magnitude_fc3_prunclass_layer_call_fn_496170a^�\]�3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_495979j�H�3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
Z__inference_prune_low_magnitude_fc3_pruned_layer_call_and_return_conditional_losses_496124lJ�HI�3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
?__inference_prune_low_magnitude_fc3_pruned_layer_call_fn_495949_�H�3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
?__inference_prune_low_magnitude_fc3_pruned_layer_call_fn_495964aJ�HI�3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
$__inference_signature_wrapper_494676�!(��4�=>�H��\�QRmnef}~uvG�D
� 
=�:
8
encoder_input'�$
encoder_input���������@"�|
@
classifier_output+�(
classifier_output���������

8
ecoder_output'�$
ecoder_output���������@