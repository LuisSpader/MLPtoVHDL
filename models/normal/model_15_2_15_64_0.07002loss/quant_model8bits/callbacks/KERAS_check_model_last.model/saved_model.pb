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
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
�
%quantize_layer_4/quantize_layer_4_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%quantize_layer_4/quantize_layer_4_min
�
9quantize_layer_4/quantize_layer_4_min/Read/ReadVariableOpReadVariableOp%quantize_layer_4/quantize_layer_4_min*
_output_shapes
: *
dtype0
�
%quantize_layer_4/quantize_layer_4_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%quantize_layer_4/quantize_layer_4_max
�
9quantize_layer_4/quantize_layer_4_max/Read/ReadVariableOpReadVariableOp%quantize_layer_4/quantize_layer_4_max*
_output_shapes
: *
dtype0
�
quantize_layer_4/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!quantize_layer_4/optimizer_step
�
3quantize_layer_4/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer_4/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_4/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_4/optimizer_step
�
0quant_dense_4/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_4/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_4/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_4/kernel_min
}
,quant_dense_4/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_4/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_4/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_4/kernel_max
}
,quant_dense_4/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_4/kernel_max*
_output_shapes
: *
dtype0
�
!quant_dense_4/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_4/post_activation_min
�
5quant_dense_4/post_activation_min/Read/ReadVariableOpReadVariableOp!quant_dense_4/post_activation_min*
_output_shapes
: *
dtype0
�
!quant_dense_4/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_4/post_activation_max
�
5quant_dense_4/post_activation_max/Read/ReadVariableOpReadVariableOp!quant_dense_4/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_5/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_5/optimizer_step
�
0quant_dense_5/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_5/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_5/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_5/kernel_min
}
,quant_dense_5/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_5/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_5/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_5/kernel_max
}
,quant_dense_5/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_5/kernel_max*
_output_shapes
: *
dtype0
�
!quant_dense_5/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_5/post_activation_min
�
5quant_dense_5/post_activation_min/Read/ReadVariableOpReadVariableOp!quant_dense_5/post_activation_min*
_output_shapes
: *
dtype0
�
!quant_dense_5/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_5/post_activation_max
�
5quant_dense_5/post_activation_max/Read/ReadVariableOpReadVariableOp!quant_dense_5/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_6/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_6/optimizer_step
�
0quant_dense_6/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_6/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_6/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_6/kernel_min
}
,quant_dense_6/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_6/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_6/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_6/kernel_max
}
,quant_dense_6/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_6/kernel_max*
_output_shapes
: *
dtype0
�
!quant_dense_6/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_6/post_activation_min
�
5quant_dense_6/post_activation_min/Read/ReadVariableOpReadVariableOp!quant_dense_6/post_activation_min*
_output_shapes
: *
dtype0
�
!quant_dense_6/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_6/post_activation_max
�
5quant_dense_6/post_activation_max/Read/ReadVariableOpReadVariableOp!quant_dense_6/post_activation_max*
_output_shapes
: *
dtype0
�
quant_dense_7/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_7/optimizer_step
�
0quant_dense_7/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_7/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_7/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_7/kernel_min
}
,quant_dense_7/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_7/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_7/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_7/kernel_max
}
,quant_dense_7/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_7/kernel_max*
_output_shapes
: *
dtype0
�
!quant_dense_7/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_7/post_activation_min
�
5quant_dense_7/post_activation_min/Read/ReadVariableOpReadVariableOp!quant_dense_7/post_activation_min*
_output_shapes
: *
dtype0
�
!quant_dense_7/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_7/post_activation_max
�
5quant_dense_7/post_activation_max/Read/ReadVariableOpReadVariableOp!quant_dense_7/post_activation_max*
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
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:@*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
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
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
�W
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
quantize_layer_4_min
quantize_layer_4_max
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
}
VARIABLE_VALUE%quantize_layer_4/quantize_layer_4_minDlayer_with_weights-0/quantize_layer_4_min/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE%quantize_layer_4/quantize_layer_4_maxDlayer_with_weights-0/quantize_layer_4_max/.ATTRIBUTES/VARIABLE_VALUE

min_var
max_var
sq
VARIABLE_VALUEquantize_layer_4/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
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
pn
VARIABLE_VALUEquant_dense_4/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

d0
hf
VARIABLE_VALUEquant_dense_4/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEquant_dense_4/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
zx
VARIABLE_VALUE!quant_dense_4/post_activation_minClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!quant_dense_4/post_activation_maxClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
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
pn
VARIABLE_VALUEquant_dense_5/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

n0
hf
VARIABLE_VALUEquant_dense_5/kernel_min:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEquant_dense_5/kernel_max:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
zx
VARIABLE_VALUE!quant_dense_5/post_activation_minClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!quant_dense_5/post_activation_maxClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
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
pn
VARIABLE_VALUEquant_dense_6/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

x0
hf
VARIABLE_VALUEquant_dense_6/kernel_min:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEquant_dense_6/kernel_max:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
zx
VARIABLE_VALUE!quant_dense_6/post_activation_minClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!quant_dense_6/post_activation_maxClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
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
pn
VARIABLE_VALUEquant_dense_7/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

�0
hf
VARIABLE_VALUEquant_dense_7/kernel_min:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEquant_dense_7/kernel_max:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
zx
VARIABLE_VALUE!quant_dense_7/post_activation_minClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!quant_dense_7/post_activation_maxClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
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
JH
VARIABLE_VALUEdense_4/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_4/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_6/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_6/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_7/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_7/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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
mk
VARIABLE_VALUEAdam/dense_4/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_4/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_5/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_5/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_6/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_6/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_7/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_7/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_4/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_4/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_5/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_5/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_6/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_6/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_7/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_7/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_5Placeholder*'
_output_shapes
:���������@*
dtype0*
shape:���������@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5%quantize_layer_4/quantize_layer_4_min%quantize_layer_4/quantize_layer_4_maxdense_4/kernelquant_dense_4/kernel_minquant_dense_4/kernel_maxdense_4/bias!quant_dense_4/post_activation_min!quant_dense_4/post_activation_maxdense_5/kernelquant_dense_5/kernel_minquant_dense_5/kernel_maxdense_5/bias!quant_dense_5/post_activation_min!quant_dense_5/post_activation_maxdense_6/kernelquant_dense_6/kernel_minquant_dense_6/kernel_maxdense_6/bias!quant_dense_6/post_activation_min!quant_dense_6/post_activation_maxdense_7/kernelquant_dense_7/kernel_minquant_dense_7/kernel_maxdense_7/bias!quant_dense_7/post_activation_min!quant_dense_7/post_activation_max*&
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
GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_444538
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename9quantize_layer_4/quantize_layer_4_min/Read/ReadVariableOp9quantize_layer_4/quantize_layer_4_max/Read/ReadVariableOp3quantize_layer_4/optimizer_step/Read/ReadVariableOp0quant_dense_4/optimizer_step/Read/ReadVariableOp,quant_dense_4/kernel_min/Read/ReadVariableOp,quant_dense_4/kernel_max/Read/ReadVariableOp5quant_dense_4/post_activation_min/Read/ReadVariableOp5quant_dense_4/post_activation_max/Read/ReadVariableOp0quant_dense_5/optimizer_step/Read/ReadVariableOp,quant_dense_5/kernel_min/Read/ReadVariableOp,quant_dense_5/kernel_max/Read/ReadVariableOp5quant_dense_5/post_activation_min/Read/ReadVariableOp5quant_dense_5/post_activation_max/Read/ReadVariableOp0quant_dense_6/optimizer_step/Read/ReadVariableOp,quant_dense_6/kernel_min/Read/ReadVariableOp,quant_dense_6/kernel_max/Read/ReadVariableOp5quant_dense_6/post_activation_min/Read/ReadVariableOp5quant_dense_6/post_activation_max/Read/ReadVariableOp0quant_dense_7/optimizer_step/Read/ReadVariableOp,quant_dense_7/kernel_min/Read/ReadVariableOp,quant_dense_7/kernel_max/Read/ReadVariableOp5quant_dense_7/post_activation_min/Read/ReadVariableOp5quant_dense_7/post_activation_max/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*C
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
GPU 2J 8� *(
f#R!
__inference__traced_save_445639
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename%quantize_layer_4/quantize_layer_4_min%quantize_layer_4/quantize_layer_4_maxquantize_layer_4/optimizer_stepquant_dense_4/optimizer_stepquant_dense_4/kernel_minquant_dense_4/kernel_max!quant_dense_4/post_activation_min!quant_dense_4/post_activation_maxquant_dense_5/optimizer_stepquant_dense_5/kernel_minquant_dense_5/kernel_max!quant_dense_5/post_activation_min!quant_dense_5/post_activation_maxquant_dense_6/optimizer_stepquant_dense_6/kernel_minquant_dense_6/kernel_max!quant_dense_6/post_activation_min!quant_dense_6/post_activation_maxquant_dense_7/optimizer_stepquant_dense_7/kernel_minquant_dense_7/kernel_max!quant_dense_7/post_activation_min!quant_dense_7/post_activation_max	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biastotalcountAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*B
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_445811��
� 
�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_445175

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
�
�
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_444987

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
�
�
1__inference_quantize_layer_4_layer_call_fn_444978

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
GPU 2J 8� *U
fPRN
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_444112o
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
� 
�
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_443538

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
�
�
(__inference_model_4_layer_call_fn_444652

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
GPU 2J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_444239o
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
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_443608

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
�
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_445287

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
�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_445398

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
�T
�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_445454

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
�
�
.__inference_quant_dense_5_layer_call_fn_445137

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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_443573o
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
�
�
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_443511

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
.__inference_quant_dense_7_layer_call_fn_445378

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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_443788o
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
� 
�
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_445063

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
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_443642

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
ƫ
�#
!__inference__wrapped_model_443495
input_5d
Zmodel_4_quantize_layer_4_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: f
\model_4_quantize_layer_4_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: f
Tmodel_4_quant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@`
Vmodel_4_quant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: `
Vmodel_4_quant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: C
5model_4_quant_dense_4_biasadd_readvariableop_resource:a
Wmodel_4_quant_dense_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: c
Ymodel_4_quant_dense_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: f
Tmodel_4_quant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:`
Vmodel_4_quant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: `
Vmodel_4_quant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: C
5model_4_quant_dense_5_biasadd_readvariableop_resource:a
Wmodel_4_quant_dense_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: c
Ymodel_4_quant_dense_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: f
Tmodel_4_quant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:`
Vmodel_4_quant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: `
Vmodel_4_quant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: C
5model_4_quant_dense_6_biasadd_readvariableop_resource:a
Wmodel_4_quant_dense_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: c
Ymodel_4_quant_dense_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: f
Tmodel_4_quant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@`
Vmodel_4_quant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: `
Vmodel_4_quant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: C
5model_4_quant_dense_7_biasadd_readvariableop_resource:@a
Wmodel_4_quant_dense_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: c
Ymodel_4_quant_dense_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��,model_4/quant_dense_4/BiasAdd/ReadVariableOp�Kmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Mmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Mmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Nmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Pmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�,model_4/quant_dense_5/BiasAdd/ReadVariableOp�Kmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Mmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Mmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Nmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Pmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�,model_4/quant_dense_6/BiasAdd/ReadVariableOp�Kmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Mmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Mmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Nmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Pmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�,model_4/quant_dense_7/BiasAdd/ReadVariableOp�Kmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Mmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Mmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Nmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Pmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Qmodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Smodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Qmodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpZmodel_4_quantize_layer_4_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Smodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp\model_4_quantize_layer_4_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Bmodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinput_5Ymodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0[model_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Kmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTmodel_4_quant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
Mmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVmodel_4_quant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Mmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpVmodel_4_quant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
<model_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsSmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Umodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Umodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
model_4/quant_dense_4/MatMulMatMulLmodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Fmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
,model_4/quant_dense_4/BiasAdd/ReadVariableOpReadVariableOp5model_4_quant_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_4/quant_dense_4/BiasAddBiasAdd&model_4/quant_dense_4/MatMul:product:04model_4/quant_dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
model_4/quant_dense_4/ReluRelu&model_4/quant_dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Nmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWmodel_4_quant_dense_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Pmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYmodel_4_quant_dense_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
?model_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars(model_4/quant_dense_4/Relu:activations:0Vmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Xmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Kmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTmodel_4_quant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Mmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVmodel_4_quant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Mmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpVmodel_4_quant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
<model_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsSmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Umodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Umodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
model_4/quant_dense_5/MatMulMatMulImodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Fmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
,model_4/quant_dense_5/BiasAdd/ReadVariableOpReadVariableOp5model_4_quant_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_4/quant_dense_5/BiasAddBiasAdd&model_4/quant_dense_5/MatMul:product:04model_4/quant_dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
model_4/quant_dense_5/ReluRelu&model_4/quant_dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Nmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWmodel_4_quant_dense_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Pmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYmodel_4_quant_dense_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
?model_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars(model_4/quant_dense_5/Relu:activations:0Vmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Xmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Kmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTmodel_4_quant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Mmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVmodel_4_quant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Mmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpVmodel_4_quant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
<model_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsSmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Umodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Umodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
model_4/quant_dense_6/MatMulMatMulImodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Fmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
,model_4/quant_dense_6/BiasAdd/ReadVariableOpReadVariableOp5model_4_quant_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_4/quant_dense_6/BiasAddBiasAdd&model_4/quant_dense_6/MatMul:product:04model_4/quant_dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
model_4/quant_dense_6/ReluRelu&model_4/quant_dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Nmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWmodel_4_quant_dense_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Pmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYmodel_4_quant_dense_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
?model_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars(model_4/quant_dense_6/Relu:activations:0Vmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Xmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Kmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTmodel_4_quant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
Mmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVmodel_4_quant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Mmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpVmodel_4_quant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
<model_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsSmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Umodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Umodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
model_4/quant_dense_7/MatMulMatMulImodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Fmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
,model_4/quant_dense_7/BiasAdd/ReadVariableOpReadVariableOp5model_4_quant_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_4/quant_dense_7/BiasAddBiasAdd&model_4/quant_dense_7/MatMul:product:04model_4/quant_dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Nmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWmodel_4_quant_dense_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Pmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpYmodel_4_quant_dense_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
?model_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars&model_4/quant_dense_7/BiasAdd:output:0Vmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Xmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityImodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp-^model_4/quant_dense_4/BiasAdd/ReadVariableOpL^model_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpN^model_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1N^model_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2O^model_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQ^model_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1-^model_4/quant_dense_5/BiasAdd/ReadVariableOpL^model_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpN^model_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1N^model_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2O^model_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQ^model_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1-^model_4/quant_dense_6/BiasAdd/ReadVariableOpL^model_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpN^model_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1N^model_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2O^model_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQ^model_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1-^model_4/quant_dense_7/BiasAdd/ReadVariableOpL^model_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpN^model_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1N^model_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2O^model_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQ^model_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1R^model_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpT^model_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,model_4/quant_dense_4/BiasAdd/ReadVariableOp,model_4/quant_dense_4/BiasAdd/ReadVariableOp2�
Kmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpKmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Mmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Mmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Mmodel_4/quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Nmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpNmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Pmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Pmodel_4/quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12\
,model_4/quant_dense_5/BiasAdd/ReadVariableOp,model_4/quant_dense_5/BiasAdd/ReadVariableOp2�
Kmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpKmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Mmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Mmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Mmodel_4/quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Nmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpNmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Pmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Pmodel_4/quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12\
,model_4/quant_dense_6/BiasAdd/ReadVariableOp,model_4/quant_dense_6/BiasAdd/ReadVariableOp2�
Kmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpKmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Mmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Mmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Mmodel_4/quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Nmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpNmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Pmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Pmodel_4/quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12\
,model_4/quant_dense_7/BiasAdd/ReadVariableOp,model_4/quant_dense_7/BiasAdd/ReadVariableOp2�
Kmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpKmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Mmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Mmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Mmodel_4/quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Nmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpNmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Pmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Pmodel_4/quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Qmodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpQmodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Smodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Smodel_4/quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_5
�U
�
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_443880

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
.__inference_quant_dense_4_layer_call_fn_445025

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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_443538o
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
�%
�	
C__inference_model_4_layer_call_and_return_conditional_losses_444473
input_5!
quantize_layer_4_444415: !
quantize_layer_4_444417: &
quant_dense_4_444420:@
quant_dense_4_444422: 
quant_dense_4_444424: "
quant_dense_4_444426:
quant_dense_4_444428: 
quant_dense_4_444430: &
quant_dense_5_444433:
quant_dense_5_444435: 
quant_dense_5_444437: "
quant_dense_5_444439:
quant_dense_5_444441: 
quant_dense_5_444443: &
quant_dense_6_444446:
quant_dense_6_444448: 
quant_dense_6_444450: "
quant_dense_6_444452:
quant_dense_6_444454: 
quant_dense_6_444456: &
quant_dense_7_444459:@
quant_dense_7_444461: 
quant_dense_7_444463: "
quant_dense_7_444465:@
quant_dense_7_444467: 
quant_dense_7_444469: 
identity��%quant_dense_4/StatefulPartitionedCall�%quant_dense_5/StatefulPartitionedCall�%quant_dense_6/StatefulPartitionedCall�%quant_dense_7/StatefulPartitionedCall�(quantize_layer_4/StatefulPartitionedCall�
(quantize_layer_4/StatefulPartitionedCallStatefulPartitionedCallinput_5quantize_layer_4_444415quantize_layer_4_444417*
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
GPU 2J 8� *U
fPRN
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_444112�
%quant_dense_4/StatefulPartitionedCallStatefulPartitionedCall1quantize_layer_4/StatefulPartitionedCall:output:0quant_dense_4_444420quant_dense_4_444422quant_dense_4_444424quant_dense_4_444426quant_dense_4_444428quant_dense_4_444430*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_444064�
%quant_dense_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_4/StatefulPartitionedCall:output:0quant_dense_5_444433quant_dense_5_444435quant_dense_5_444437quant_dense_5_444439quant_dense_5_444441quant_dense_5_444443*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_443972�
%quant_dense_6/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_5/StatefulPartitionedCall:output:0quant_dense_6_444446quant_dense_6_444448quant_dense_6_444450quant_dense_6_444452quant_dense_6_444454quant_dense_6_444456*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_443880�
%quant_dense_7/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_6/StatefulPartitionedCall:output:0quant_dense_7_444459quant_dense_7_444461quant_dense_7_444463quant_dense_7_444465quant_dense_7_444467quant_dense_7_444469*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_443788}
IdentityIdentity.quant_dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp&^quant_dense_4/StatefulPartitionedCall&^quant_dense_5/StatefulPartitionedCall&^quant_dense_6/StatefulPartitionedCall&^quant_dense_7/StatefulPartitionedCall)^quantize_layer_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%quant_dense_4/StatefulPartitionedCall%quant_dense_4/StatefulPartitionedCall2N
%quant_dense_5/StatefulPartitionedCall%quant_dense_5/StatefulPartitionedCall2N
%quant_dense_6/StatefulPartitionedCall%quant_dense_6/StatefulPartitionedCall2N
%quant_dense_7/StatefulPartitionedCall%quant_dense_7/StatefulPartitionedCall2T
(quantize_layer_4/StatefulPartitionedCall(quantize_layer_4/StatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_5
�U
�
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_445120

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
� 
�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_443573

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
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_445344

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
�h
�
__inference__traced_save_445639
file_prefixD
@savev2_quantize_layer_4_quantize_layer_4_min_read_readvariableopD
@savev2_quantize_layer_4_quantize_layer_4_max_read_readvariableop>
:savev2_quantize_layer_4_optimizer_step_read_readvariableop;
7savev2_quant_dense_4_optimizer_step_read_readvariableop7
3savev2_quant_dense_4_kernel_min_read_readvariableop7
3savev2_quant_dense_4_kernel_max_read_readvariableop@
<savev2_quant_dense_4_post_activation_min_read_readvariableop@
<savev2_quant_dense_4_post_activation_max_read_readvariableop;
7savev2_quant_dense_5_optimizer_step_read_readvariableop7
3savev2_quant_dense_5_kernel_min_read_readvariableop7
3savev2_quant_dense_5_kernel_max_read_readvariableop@
<savev2_quant_dense_5_post_activation_min_read_readvariableop@
<savev2_quant_dense_5_post_activation_max_read_readvariableop;
7savev2_quant_dense_6_optimizer_step_read_readvariableop7
3savev2_quant_dense_6_kernel_min_read_readvariableop7
3savev2_quant_dense_6_kernel_max_read_readvariableop@
<savev2_quant_dense_6_post_activation_min_read_readvariableop@
<savev2_quant_dense_6_post_activation_max_read_readvariableop;
7savev2_quant_dense_7_optimizer_step_read_readvariableop7
3savev2_quant_dense_7_kernel_min_read_readvariableop7
3savev2_quant_dense_7_kernel_max_read_readvariableop@
<savev2_quant_dense_7_post_activation_min_read_readvariableop@
<savev2_quant_dense_7_post_activation_max_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
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
value�B�7BDlayer_with_weights-0/quantize_layer_4_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-0/quantize_layer_4_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0@savev2_quantize_layer_4_quantize_layer_4_min_read_readvariableop@savev2_quantize_layer_4_quantize_layer_4_max_read_readvariableop:savev2_quantize_layer_4_optimizer_step_read_readvariableop7savev2_quant_dense_4_optimizer_step_read_readvariableop3savev2_quant_dense_4_kernel_min_read_readvariableop3savev2_quant_dense_4_kernel_max_read_readvariableop<savev2_quant_dense_4_post_activation_min_read_readvariableop<savev2_quant_dense_4_post_activation_max_read_readvariableop7savev2_quant_dense_5_optimizer_step_read_readvariableop3savev2_quant_dense_5_kernel_min_read_readvariableop3savev2_quant_dense_5_kernel_max_read_readvariableop<savev2_quant_dense_5_post_activation_min_read_readvariableop<savev2_quant_dense_5_post_activation_max_read_readvariableop7savev2_quant_dense_6_optimizer_step_read_readvariableop3savev2_quant_dense_6_kernel_min_read_readvariableop3savev2_quant_dense_6_kernel_max_read_readvariableop<savev2_quant_dense_6_post_activation_min_read_readvariableop<savev2_quant_dense_6_post_activation_max_read_readvariableop7savev2_quant_dense_7_optimizer_step_read_readvariableop3savev2_quant_dense_7_kernel_min_read_readvariableop3savev2_quant_dense_7_kernel_max_read_readvariableop<savev2_quant_dense_7_post_activation_min_read_readvariableop<savev2_quant_dense_7_post_activation_max_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�%
�	
C__inference_model_4_layer_call_and_return_conditional_losses_443657

inputs!
quantize_layer_4_443512: !
quantize_layer_4_443514: &
quant_dense_4_443539:@
quant_dense_4_443541: 
quant_dense_4_443543: "
quant_dense_4_443545:
quant_dense_4_443547: 
quant_dense_4_443549: &
quant_dense_5_443574:
quant_dense_5_443576: 
quant_dense_5_443578: "
quant_dense_5_443580:
quant_dense_5_443582: 
quant_dense_5_443584: &
quant_dense_6_443609:
quant_dense_6_443611: 
quant_dense_6_443613: "
quant_dense_6_443615:
quant_dense_6_443617: 
quant_dense_6_443619: &
quant_dense_7_443643:@
quant_dense_7_443645: 
quant_dense_7_443647: "
quant_dense_7_443649:@
quant_dense_7_443651: 
quant_dense_7_443653: 
identity��%quant_dense_4/StatefulPartitionedCall�%quant_dense_5/StatefulPartitionedCall�%quant_dense_6/StatefulPartitionedCall�%quant_dense_7/StatefulPartitionedCall�(quantize_layer_4/StatefulPartitionedCall�
(quantize_layer_4/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_4_443512quantize_layer_4_443514*
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
GPU 2J 8� *U
fPRN
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_443511�
%quant_dense_4/StatefulPartitionedCallStatefulPartitionedCall1quantize_layer_4/StatefulPartitionedCall:output:0quant_dense_4_443539quant_dense_4_443541quant_dense_4_443543quant_dense_4_443545quant_dense_4_443547quant_dense_4_443549*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_443538�
%quant_dense_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_4/StatefulPartitionedCall:output:0quant_dense_5_443574quant_dense_5_443576quant_dense_5_443578quant_dense_5_443580quant_dense_5_443582quant_dense_5_443584*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_443573�
%quant_dense_6/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_5/StatefulPartitionedCall:output:0quant_dense_6_443609quant_dense_6_443611quant_dense_6_443613quant_dense_6_443615quant_dense_6_443617quant_dense_6_443619*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_443608�
%quant_dense_7/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_6/StatefulPartitionedCall:output:0quant_dense_7_443643quant_dense_7_443645quant_dense_7_443647quant_dense_7_443649quant_dense_7_443651quant_dense_7_443653*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_443642}
IdentityIdentity.quant_dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp&^quant_dense_4/StatefulPartitionedCall&^quant_dense_5/StatefulPartitionedCall&^quant_dense_6/StatefulPartitionedCall&^quant_dense_7/StatefulPartitionedCall)^quantize_layer_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%quant_dense_4/StatefulPartitionedCall%quant_dense_4/StatefulPartitionedCall2N
%quant_dense_5/StatefulPartitionedCall%quant_dense_5/StatefulPartitionedCall2N
%quant_dense_6/StatefulPartitionedCall%quant_dense_6/StatefulPartitionedCall2N
%quant_dense_7/StatefulPartitionedCall%quant_dense_7/StatefulPartitionedCall2T
(quantize_layer_4/StatefulPartitionedCall(quantize_layer_4/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_5_layer_call_fn_445154

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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_443972o
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
��
�-
C__inference_model_4_layer_call_and_return_conditional_losses_444960

inputsL
Bquantize_layer_4_allvaluesquantize_minimum_readvariableop_resource: L
Bquantize_layer_4_allvaluesquantize_maximum_readvariableop_resource: K
9quant_dense_4_lastvaluequant_rank_readvariableop_resource:@=
3quant_dense_4_lastvaluequant_assignminlast_resource: =
3quant_dense_4_lastvaluequant_assignmaxlast_resource: ;
-quant_dense_4_biasadd_readvariableop_resource:N
Dquant_dense_4_movingavgquantize_assignminema_readvariableop_resource: N
Dquant_dense_4_movingavgquantize_assignmaxema_readvariableop_resource: K
9quant_dense_5_lastvaluequant_rank_readvariableop_resource:=
3quant_dense_5_lastvaluequant_assignminlast_resource: =
3quant_dense_5_lastvaluequant_assignmaxlast_resource: ;
-quant_dense_5_biasadd_readvariableop_resource:N
Dquant_dense_5_movingavgquantize_assignminema_readvariableop_resource: N
Dquant_dense_5_movingavgquantize_assignmaxema_readvariableop_resource: K
9quant_dense_6_lastvaluequant_rank_readvariableop_resource:=
3quant_dense_6_lastvaluequant_assignminlast_resource: =
3quant_dense_6_lastvaluequant_assignmaxlast_resource: ;
-quant_dense_6_biasadd_readvariableop_resource:N
Dquant_dense_6_movingavgquantize_assignminema_readvariableop_resource: N
Dquant_dense_6_movingavgquantize_assignmaxema_readvariableop_resource: K
9quant_dense_7_lastvaluequant_rank_readvariableop_resource:@=
3quant_dense_7_lastvaluequant_assignminlast_resource: =
3quant_dense_7_lastvaluequant_assignmaxlast_resource: ;
-quant_dense_7_biasadd_readvariableop_resource:@N
Dquant_dense_7_movingavgquantize_assignminema_readvariableop_resource: N
Dquant_dense_7_movingavgquantize_assignmaxema_readvariableop_resource: 
identity��$quant_dense_4/BiasAdd/ReadVariableOp�*quant_dense_4/LastValueQuant/AssignMaxLast�*quant_dense_4/LastValueQuant/AssignMinLast�4quant_dense_4/LastValueQuant/BatchMax/ReadVariableOp�4quant_dense_4/LastValueQuant/BatchMin/ReadVariableOp�Cquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�@quant_dense_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�;quant_dense_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�@quant_dense_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�;quant_dense_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Fquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�$quant_dense_5/BiasAdd/ReadVariableOp�*quant_dense_5/LastValueQuant/AssignMaxLast�*quant_dense_5/LastValueQuant/AssignMinLast�4quant_dense_5/LastValueQuant/BatchMax/ReadVariableOp�4quant_dense_5/LastValueQuant/BatchMin/ReadVariableOp�Cquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�@quant_dense_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�;quant_dense_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�@quant_dense_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�;quant_dense_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Fquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�$quant_dense_6/BiasAdd/ReadVariableOp�*quant_dense_6/LastValueQuant/AssignMaxLast�*quant_dense_6/LastValueQuant/AssignMinLast�4quant_dense_6/LastValueQuant/BatchMax/ReadVariableOp�4quant_dense_6/LastValueQuant/BatchMin/ReadVariableOp�Cquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�@quant_dense_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�;quant_dense_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�@quant_dense_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�;quant_dense_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Fquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�$quant_dense_7/BiasAdd/ReadVariableOp�*quant_dense_7/LastValueQuant/AssignMaxLast�*quant_dense_7/LastValueQuant/AssignMinLast�4quant_dense_7/LastValueQuant/BatchMax/ReadVariableOp�4quant_dense_7/LastValueQuant/BatchMin/ReadVariableOp�Cquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�@quant_dense_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�;quant_dense_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�@quant_dense_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�;quant_dense_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Fquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�4quantize_layer_4/AllValuesQuantize/AssignMaxAllValue�4quantize_layer_4/AllValuesQuantize/AssignMinAllValue�Iquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Kquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�9quantize_layer_4/AllValuesQuantize/Maximum/ReadVariableOp�9quantize_layer_4/AllValuesQuantize/Minimum/ReadVariableOpy
(quantize_layer_4/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
+quantize_layer_4/AllValuesQuantize/BatchMinMininputs1quantize_layer_4/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: {
*quantize_layer_4/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
+quantize_layer_4/AllValuesQuantize/BatchMaxMaxinputs3quantize_layer_4/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
9quantize_layer_4/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOpBquantize_layer_4_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
*quantize_layer_4/AllValuesQuantize/MinimumMinimumAquantize_layer_4/AllValuesQuantize/Minimum/ReadVariableOp:value:04quantize_layer_4/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: s
.quantize_layer_4/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quantize_layer_4/AllValuesQuantize/Minimum_1Minimum.quantize_layer_4/AllValuesQuantize/Minimum:z:07quantize_layer_4/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
9quantize_layer_4/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOpBquantize_layer_4_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
*quantize_layer_4/AllValuesQuantize/MaximumMaximumAquantize_layer_4/AllValuesQuantize/Maximum/ReadVariableOp:value:04quantize_layer_4/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: s
.quantize_layer_4/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quantize_layer_4/AllValuesQuantize/Maximum_1Maximum.quantize_layer_4/AllValuesQuantize/Maximum:z:07quantize_layer_4/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
4quantize_layer_4/AllValuesQuantize/AssignMinAllValueAssignVariableOpBquantize_layer_4_allvaluesquantize_minimum_readvariableop_resource0quantize_layer_4/AllValuesQuantize/Minimum_1:z:0:^quantize_layer_4/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0�
4quantize_layer_4/AllValuesQuantize/AssignMaxAllValueAssignVariableOpBquantize_layer_4_allvaluesquantize_maximum_readvariableop_resource0quantize_layer_4/AllValuesQuantize/Maximum_1:z:0:^quantize_layer_4/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0�
Iquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpBquantize_layer_4_allvaluesquantize_minimum_readvariableop_resource5^quantize_layer_4/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
Kquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpBquantize_layer_4_allvaluesquantize_maximum_readvariableop_resource5^quantize_layer_4/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
:quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsQquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Squantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
0quant_dense_4/LastValueQuant/Rank/ReadVariableOpReadVariableOp9quant_dense_4_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0c
!quant_dense_4/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :j
(quant_dense_4/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(quant_dense_4/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"quant_dense_4/LastValueQuant/rangeRange1quant_dense_4/LastValueQuant/range/start:output:0*quant_dense_4/LastValueQuant/Rank:output:01quant_dense_4/LastValueQuant/range/delta:output:0*
_output_shapes
:�
4quant_dense_4/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp9quant_dense_4_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
%quant_dense_4/LastValueQuant/BatchMinMin<quant_dense_4/LastValueQuant/BatchMin/ReadVariableOp:value:0+quant_dense_4/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
2quant_dense_4/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp9quant_dense_4_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0e
#quant_dense_4/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :l
*quant_dense_4/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : l
*quant_dense_4/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
$quant_dense_4/LastValueQuant/range_1Range3quant_dense_4/LastValueQuant/range_1/start:output:0,quant_dense_4/LastValueQuant/Rank_1:output:03quant_dense_4/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
4quant_dense_4/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp9quant_dense_4_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
%quant_dense_4/LastValueQuant/BatchMaxMax<quant_dense_4/LastValueQuant/BatchMax/ReadVariableOp:value:0-quant_dense_4/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: k
&quant_dense_4/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
$quant_dense_4/LastValueQuant/truedivRealDiv.quant_dense_4/LastValueQuant/BatchMax:output:0/quant_dense_4/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
$quant_dense_4/LastValueQuant/MinimumMinimum.quant_dense_4/LastValueQuant/BatchMin:output:0(quant_dense_4/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: g
"quant_dense_4/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
 quant_dense_4/LastValueQuant/mulMul.quant_dense_4/LastValueQuant/BatchMin:output:0+quant_dense_4/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
$quant_dense_4/LastValueQuant/MaximumMaximum.quant_dense_4/LastValueQuant/BatchMax:output:0$quant_dense_4/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
*quant_dense_4/LastValueQuant/AssignMinLastAssignVariableOp3quant_dense_4_lastvaluequant_assignminlast_resource(quant_dense_4/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
*quant_dense_4/LastValueQuant/AssignMaxLastAssignVariableOp3quant_dense_4_lastvaluequant_assignmaxlast_resource(quant_dense_4/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Cquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9quant_dense_4_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp3quant_dense_4_lastvaluequant_assignminlast_resource+^quant_dense_4/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp3quant_dense_4_lastvaluequant_assignmaxlast_resource+^quant_dense_4/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
4quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
quant_dense_4/MatMulMatMulDquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
$quant_dense_4/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_4/BiasAddBiasAddquant_dense_4/MatMul:product:0,quant_dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
quant_dense_4/ReluReluquant_dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������v
%quant_dense_4/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_4/MovingAvgQuantize/BatchMinMin quant_dense_4/Relu:activations:0.quant_dense_4/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: x
'quant_dense_4/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_4/MovingAvgQuantize/BatchMaxMax quant_dense_4/Relu:activations:00quant_dense_4/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: n
)quant_dense_4/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_4/MovingAvgQuantize/MinimumMinimum1quant_dense_4/MovingAvgQuantize/BatchMin:output:02quant_dense_4/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: n
)quant_dense_4/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_4/MovingAvgQuantize/MaximumMaximum1quant_dense_4/MovingAvgQuantize/BatchMax:output:02quant_dense_4/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: w
2quant_dense_4/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_dense_4_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_4/MovingAvgQuantize/AssignMinEma/subSubCquant_dense_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_dense_4/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
0quant_dense_4/MovingAvgQuantize/AssignMinEma/mulMul4quant_dense_4/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_dense_4/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_4_movingavgquantize_assignminema_readvariableop_resource4quant_dense_4/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_dense_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0w
2quant_dense_4/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_dense_4_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_4/MovingAvgQuantize/AssignMaxEma/subSubCquant_dense_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_dense_4/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
0quant_dense_4/MovingAvgQuantize/AssignMaxEma/mulMul4quant_dense_4/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_dense_4/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_4_movingavgquantize_assignmaxema_readvariableop_resource4quant_dense_4/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_dense_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Fquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_dense_4_movingavgquantize_assignminema_readvariableop_resourceA^quant_dense_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Hquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_dense_4_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_dense_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
7quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_dense_4/Relu:activations:0Nquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
0quant_dense_5/LastValueQuant/Rank/ReadVariableOpReadVariableOp9quant_dense_5_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0c
!quant_dense_5/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :j
(quant_dense_5/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(quant_dense_5/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"quant_dense_5/LastValueQuant/rangeRange1quant_dense_5/LastValueQuant/range/start:output:0*quant_dense_5/LastValueQuant/Rank:output:01quant_dense_5/LastValueQuant/range/delta:output:0*
_output_shapes
:�
4quant_dense_5/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp9quant_dense_5_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
%quant_dense_5/LastValueQuant/BatchMinMin<quant_dense_5/LastValueQuant/BatchMin/ReadVariableOp:value:0+quant_dense_5/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
2quant_dense_5/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp9quant_dense_5_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0e
#quant_dense_5/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :l
*quant_dense_5/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : l
*quant_dense_5/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
$quant_dense_5/LastValueQuant/range_1Range3quant_dense_5/LastValueQuant/range_1/start:output:0,quant_dense_5/LastValueQuant/Rank_1:output:03quant_dense_5/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
4quant_dense_5/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp9quant_dense_5_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
%quant_dense_5/LastValueQuant/BatchMaxMax<quant_dense_5/LastValueQuant/BatchMax/ReadVariableOp:value:0-quant_dense_5/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: k
&quant_dense_5/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
$quant_dense_5/LastValueQuant/truedivRealDiv.quant_dense_5/LastValueQuant/BatchMax:output:0/quant_dense_5/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
$quant_dense_5/LastValueQuant/MinimumMinimum.quant_dense_5/LastValueQuant/BatchMin:output:0(quant_dense_5/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: g
"quant_dense_5/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
 quant_dense_5/LastValueQuant/mulMul.quant_dense_5/LastValueQuant/BatchMin:output:0+quant_dense_5/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
$quant_dense_5/LastValueQuant/MaximumMaximum.quant_dense_5/LastValueQuant/BatchMax:output:0$quant_dense_5/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
*quant_dense_5/LastValueQuant/AssignMinLastAssignVariableOp3quant_dense_5_lastvaluequant_assignminlast_resource(quant_dense_5/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
*quant_dense_5/LastValueQuant/AssignMaxLastAssignVariableOp3quant_dense_5_lastvaluequant_assignmaxlast_resource(quant_dense_5/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Cquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9quant_dense_5_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp3quant_dense_5_lastvaluequant_assignminlast_resource+^quant_dense_5/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp3quant_dense_5_lastvaluequant_assignmaxlast_resource+^quant_dense_5/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
4quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_5/MatMulMatMulAquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
$quant_dense_5/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_5/BiasAddBiasAddquant_dense_5/MatMul:product:0,quant_dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
quant_dense_5/ReluReluquant_dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������v
%quant_dense_5/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_5/MovingAvgQuantize/BatchMinMin quant_dense_5/Relu:activations:0.quant_dense_5/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: x
'quant_dense_5/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_5/MovingAvgQuantize/BatchMaxMax quant_dense_5/Relu:activations:00quant_dense_5/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: n
)quant_dense_5/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_5/MovingAvgQuantize/MinimumMinimum1quant_dense_5/MovingAvgQuantize/BatchMin:output:02quant_dense_5/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: n
)quant_dense_5/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_5/MovingAvgQuantize/MaximumMaximum1quant_dense_5/MovingAvgQuantize/BatchMax:output:02quant_dense_5/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: w
2quant_dense_5/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_5/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_dense_5_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_5/MovingAvgQuantize/AssignMinEma/subSubCquant_dense_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_dense_5/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
0quant_dense_5/MovingAvgQuantize/AssignMinEma/mulMul4quant_dense_5/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_dense_5/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_5_movingavgquantize_assignminema_readvariableop_resource4quant_dense_5/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_dense_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0w
2quant_dense_5/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_dense_5_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_5/MovingAvgQuantize/AssignMaxEma/subSubCquant_dense_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_dense_5/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
0quant_dense_5/MovingAvgQuantize/AssignMaxEma/mulMul4quant_dense_5/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_dense_5/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_5_movingavgquantize_assignmaxema_readvariableop_resource4quant_dense_5/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_dense_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Fquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_dense_5_movingavgquantize_assignminema_readvariableop_resourceA^quant_dense_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Hquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_dense_5_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_dense_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
7quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_dense_5/Relu:activations:0Nquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
0quant_dense_6/LastValueQuant/Rank/ReadVariableOpReadVariableOp9quant_dense_6_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0c
!quant_dense_6/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :j
(quant_dense_6/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(quant_dense_6/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"quant_dense_6/LastValueQuant/rangeRange1quant_dense_6/LastValueQuant/range/start:output:0*quant_dense_6/LastValueQuant/Rank:output:01quant_dense_6/LastValueQuant/range/delta:output:0*
_output_shapes
:�
4quant_dense_6/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp9quant_dense_6_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
%quant_dense_6/LastValueQuant/BatchMinMin<quant_dense_6/LastValueQuant/BatchMin/ReadVariableOp:value:0+quant_dense_6/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
2quant_dense_6/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp9quant_dense_6_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0e
#quant_dense_6/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :l
*quant_dense_6/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : l
*quant_dense_6/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
$quant_dense_6/LastValueQuant/range_1Range3quant_dense_6/LastValueQuant/range_1/start:output:0,quant_dense_6/LastValueQuant/Rank_1:output:03quant_dense_6/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
4quant_dense_6/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp9quant_dense_6_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
%quant_dense_6/LastValueQuant/BatchMaxMax<quant_dense_6/LastValueQuant/BatchMax/ReadVariableOp:value:0-quant_dense_6/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: k
&quant_dense_6/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
$quant_dense_6/LastValueQuant/truedivRealDiv.quant_dense_6/LastValueQuant/BatchMax:output:0/quant_dense_6/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
$quant_dense_6/LastValueQuant/MinimumMinimum.quant_dense_6/LastValueQuant/BatchMin:output:0(quant_dense_6/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: g
"quant_dense_6/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
 quant_dense_6/LastValueQuant/mulMul.quant_dense_6/LastValueQuant/BatchMin:output:0+quant_dense_6/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
$quant_dense_6/LastValueQuant/MaximumMaximum.quant_dense_6/LastValueQuant/BatchMax:output:0$quant_dense_6/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
*quant_dense_6/LastValueQuant/AssignMinLastAssignVariableOp3quant_dense_6_lastvaluequant_assignminlast_resource(quant_dense_6/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
*quant_dense_6/LastValueQuant/AssignMaxLastAssignVariableOp3quant_dense_6_lastvaluequant_assignmaxlast_resource(quant_dense_6/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Cquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9quant_dense_6_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:*
dtype0�
Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp3quant_dense_6_lastvaluequant_assignminlast_resource+^quant_dense_6/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp3quant_dense_6_lastvaluequant_assignmaxlast_resource+^quant_dense_6/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
4quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_6/MatMulMatMulAquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
$quant_dense_6/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_6/BiasAddBiasAddquant_dense_6/MatMul:product:0,quant_dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
quant_dense_6/ReluReluquant_dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������v
%quant_dense_6/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_6/MovingAvgQuantize/BatchMinMin quant_dense_6/Relu:activations:0.quant_dense_6/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: x
'quant_dense_6/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_6/MovingAvgQuantize/BatchMaxMax quant_dense_6/Relu:activations:00quant_dense_6/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: n
)quant_dense_6/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_6/MovingAvgQuantize/MinimumMinimum1quant_dense_6/MovingAvgQuantize/BatchMin:output:02quant_dense_6/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: n
)quant_dense_6/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_6/MovingAvgQuantize/MaximumMaximum1quant_dense_6/MovingAvgQuantize/BatchMax:output:02quant_dense_6/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: w
2quant_dense_6/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_dense_6_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_6/MovingAvgQuantize/AssignMinEma/subSubCquant_dense_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_dense_6/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
0quant_dense_6/MovingAvgQuantize/AssignMinEma/mulMul4quant_dense_6/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_dense_6/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_6_movingavgquantize_assignminema_readvariableop_resource4quant_dense_6/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_dense_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0w
2quant_dense_6/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_dense_6_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_6/MovingAvgQuantize/AssignMaxEma/subSubCquant_dense_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_dense_6/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
0quant_dense_6/MovingAvgQuantize/AssignMaxEma/mulMul4quant_dense_6/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_dense_6/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_6_movingavgquantize_assignmaxema_readvariableop_resource4quant_dense_6/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_dense_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Fquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_dense_6_movingavgquantize_assignminema_readvariableop_resourceA^quant_dense_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Hquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_dense_6_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_dense_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
7quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_dense_6/Relu:activations:0Nquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
0quant_dense_7/LastValueQuant/Rank/ReadVariableOpReadVariableOp9quant_dense_7_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0c
!quant_dense_7/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :j
(quant_dense_7/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(quant_dense_7/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"quant_dense_7/LastValueQuant/rangeRange1quant_dense_7/LastValueQuant/range/start:output:0*quant_dense_7/LastValueQuant/Rank:output:01quant_dense_7/LastValueQuant/range/delta:output:0*
_output_shapes
:�
4quant_dense_7/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp9quant_dense_7_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
%quant_dense_7/LastValueQuant/BatchMinMin<quant_dense_7/LastValueQuant/BatchMin/ReadVariableOp:value:0+quant_dense_7/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
2quant_dense_7/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp9quant_dense_7_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0e
#quant_dense_7/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :l
*quant_dense_7/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : l
*quant_dense_7/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
$quant_dense_7/LastValueQuant/range_1Range3quant_dense_7/LastValueQuant/range_1/start:output:0,quant_dense_7/LastValueQuant/Rank_1:output:03quant_dense_7/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
4quant_dense_7/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp9quant_dense_7_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
%quant_dense_7/LastValueQuant/BatchMaxMax<quant_dense_7/LastValueQuant/BatchMax/ReadVariableOp:value:0-quant_dense_7/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: k
&quant_dense_7/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
$quant_dense_7/LastValueQuant/truedivRealDiv.quant_dense_7/LastValueQuant/BatchMax:output:0/quant_dense_7/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
$quant_dense_7/LastValueQuant/MinimumMinimum.quant_dense_7/LastValueQuant/BatchMin:output:0(quant_dense_7/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: g
"quant_dense_7/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
 quant_dense_7/LastValueQuant/mulMul.quant_dense_7/LastValueQuant/BatchMin:output:0+quant_dense_7/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
$quant_dense_7/LastValueQuant/MaximumMaximum.quant_dense_7/LastValueQuant/BatchMax:output:0$quant_dense_7/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
*quant_dense_7/LastValueQuant/AssignMinLastAssignVariableOp3quant_dense_7_lastvaluequant_assignminlast_resource(quant_dense_7/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
*quant_dense_7/LastValueQuant/AssignMaxLastAssignVariableOp3quant_dense_7_lastvaluequant_assignmaxlast_resource(quant_dense_7/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Cquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9quant_dense_7_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@*
dtype0�
Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp3quant_dense_7_lastvaluequant_assignminlast_resource+^quant_dense_7/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp3quant_dense_7_lastvaluequant_assignmaxlast_resource+^quant_dense_7/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
4quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
quant_dense_7/MatMulMatMulAquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
$quant_dense_7/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_7/BiasAddBiasAddquant_dense_7/MatMul:product:0,quant_dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@v
%quant_dense_7/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_7/MovingAvgQuantize/BatchMinMinquant_dense_7/BiasAdd:output:0.quant_dense_7/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: x
'quant_dense_7/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_7/MovingAvgQuantize/BatchMaxMaxquant_dense_7/BiasAdd:output:00quant_dense_7/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: n
)quant_dense_7/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_7/MovingAvgQuantize/MinimumMinimum1quant_dense_7/MovingAvgQuantize/BatchMin:output:02quant_dense_7/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: n
)quant_dense_7/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_7/MovingAvgQuantize/MaximumMaximum1quant_dense_7/MovingAvgQuantize/BatchMax:output:02quant_dense_7/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: w
2quant_dense_7/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_7/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_dense_7_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_7/MovingAvgQuantize/AssignMinEma/subSubCquant_dense_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_dense_7/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
0quant_dense_7/MovingAvgQuantize/AssignMinEma/mulMul4quant_dense_7/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_dense_7/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_7_movingavgquantize_assignminema_readvariableop_resource4quant_dense_7/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_dense_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0w
2quant_dense_7/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_dense_7_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_7/MovingAvgQuantize/AssignMaxEma/subSubCquant_dense_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_dense_7/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
0quant_dense_7/MovingAvgQuantize/AssignMaxEma/mulMul4quant_dense_7/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_dense_7/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_7_movingavgquantize_assignmaxema_readvariableop_resource4quant_dense_7/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_dense_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Fquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_dense_7_movingavgquantize_assignminema_readvariableop_resourceA^quant_dense_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Hquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_dense_7_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_dense_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
7quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_7/BiasAdd:output:0Nquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityAquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp%^quant_dense_4/BiasAdd/ReadVariableOp+^quant_dense_4/LastValueQuant/AssignMaxLast+^quant_dense_4/LastValueQuant/AssignMinLast5^quant_dense_4/LastValueQuant/BatchMax/ReadVariableOp5^quant_dense_4/LastValueQuant/BatchMin/ReadVariableOpD^quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2A^quant_dense_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<^quant_dense_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpA^quant_dense_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<^quant_dense_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpG^quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_5/BiasAdd/ReadVariableOp+^quant_dense_5/LastValueQuant/AssignMaxLast+^quant_dense_5/LastValueQuant/AssignMinLast5^quant_dense_5/LastValueQuant/BatchMax/ReadVariableOp5^quant_dense_5/LastValueQuant/BatchMin/ReadVariableOpD^quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2A^quant_dense_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<^quant_dense_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOpA^quant_dense_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<^quant_dense_5/MovingAvgQuantize/AssignMinEma/ReadVariableOpG^quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_6/BiasAdd/ReadVariableOp+^quant_dense_6/LastValueQuant/AssignMaxLast+^quant_dense_6/LastValueQuant/AssignMinLast5^quant_dense_6/LastValueQuant/BatchMax/ReadVariableOp5^quant_dense_6/LastValueQuant/BatchMin/ReadVariableOpD^quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2A^quant_dense_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<^quant_dense_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpA^quant_dense_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<^quant_dense_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpG^quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_7/BiasAdd/ReadVariableOp+^quant_dense_7/LastValueQuant/AssignMaxLast+^quant_dense_7/LastValueQuant/AssignMinLast5^quant_dense_7/LastValueQuant/BatchMax/ReadVariableOp5^quant_dense_7/LastValueQuant/BatchMin/ReadVariableOpD^quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2A^quant_dense_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<^quant_dense_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOpA^quant_dense_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<^quant_dense_7/MovingAvgQuantize/AssignMinEma/ReadVariableOpG^quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_15^quantize_layer_4/AllValuesQuantize/AssignMaxAllValue5^quantize_layer_4/AllValuesQuantize/AssignMinAllValueJ^quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpL^quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:^quantize_layer_4/AllValuesQuantize/Maximum/ReadVariableOp:^quantize_layer_4/AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$quant_dense_4/BiasAdd/ReadVariableOp$quant_dense_4/BiasAdd/ReadVariableOp2X
*quant_dense_4/LastValueQuant/AssignMaxLast*quant_dense_4/LastValueQuant/AssignMaxLast2X
*quant_dense_4/LastValueQuant/AssignMinLast*quant_dense_4/LastValueQuant/AssignMinLast2l
4quant_dense_4/LastValueQuant/BatchMax/ReadVariableOp4quant_dense_4/LastValueQuant/BatchMax/ReadVariableOp2l
4quant_dense_4/LastValueQuant/BatchMin/ReadVariableOp4quant_dense_4/LastValueQuant/BatchMin/ReadVariableOp2�
Cquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
@quant_dense_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_dense_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2z
;quant_dense_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp;quant_dense_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
@quant_dense_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_dense_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2z
;quant_dense_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp;quant_dense_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Fquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_5/BiasAdd/ReadVariableOp$quant_dense_5/BiasAdd/ReadVariableOp2X
*quant_dense_5/LastValueQuant/AssignMaxLast*quant_dense_5/LastValueQuant/AssignMaxLast2X
*quant_dense_5/LastValueQuant/AssignMinLast*quant_dense_5/LastValueQuant/AssignMinLast2l
4quant_dense_5/LastValueQuant/BatchMax/ReadVariableOp4quant_dense_5/LastValueQuant/BatchMax/ReadVariableOp2l
4quant_dense_5/LastValueQuant/BatchMin/ReadVariableOp4quant_dense_5/LastValueQuant/BatchMin/ReadVariableOp2�
Cquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
@quant_dense_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_dense_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2z
;quant_dense_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp;quant_dense_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
@quant_dense_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_dense_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2z
;quant_dense_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp;quant_dense_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Fquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_6/BiasAdd/ReadVariableOp$quant_dense_6/BiasAdd/ReadVariableOp2X
*quant_dense_6/LastValueQuant/AssignMaxLast*quant_dense_6/LastValueQuant/AssignMaxLast2X
*quant_dense_6/LastValueQuant/AssignMinLast*quant_dense_6/LastValueQuant/AssignMinLast2l
4quant_dense_6/LastValueQuant/BatchMax/ReadVariableOp4quant_dense_6/LastValueQuant/BatchMax/ReadVariableOp2l
4quant_dense_6/LastValueQuant/BatchMin/ReadVariableOp4quant_dense_6/LastValueQuant/BatchMin/ReadVariableOp2�
Cquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
@quant_dense_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_dense_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2z
;quant_dense_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp;quant_dense_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
@quant_dense_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_dense_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2z
;quant_dense_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp;quant_dense_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Fquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_7/BiasAdd/ReadVariableOp$quant_dense_7/BiasAdd/ReadVariableOp2X
*quant_dense_7/LastValueQuant/AssignMaxLast*quant_dense_7/LastValueQuant/AssignMaxLast2X
*quant_dense_7/LastValueQuant/AssignMinLast*quant_dense_7/LastValueQuant/AssignMinLast2l
4quant_dense_7/LastValueQuant/BatchMax/ReadVariableOp4quant_dense_7/LastValueQuant/BatchMax/ReadVariableOp2l
4quant_dense_7/LastValueQuant/BatchMin/ReadVariableOp4quant_dense_7/LastValueQuant/BatchMin/ReadVariableOp2�
Cquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
@quant_dense_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_dense_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2z
;quant_dense_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp;quant_dense_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
@quant_dense_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_dense_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2z
;quant_dense_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp;quant_dense_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Fquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12l
4quantize_layer_4/AllValuesQuantize/AssignMaxAllValue4quantize_layer_4/AllValuesQuantize/AssignMaxAllValue2l
4quantize_layer_4/AllValuesQuantize/AssignMinAllValue4quantize_layer_4/AllValuesQuantize/AssignMinAllValue2�
Iquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpIquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Kquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Kquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12v
9quantize_layer_4/AllValuesQuantize/Maximum/ReadVariableOp9quantize_layer_4/AllValuesQuantize/Maximum/ReadVariableOp2v
9quantize_layer_4/AllValuesQuantize/Minimum/ReadVariableOp9quantize_layer_4/AllValuesQuantize/Minimum/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�U
�
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_444064

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
�
�
.__inference_quant_dense_6_layer_call_fn_445249

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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_443608o
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
�
�
(__inference_model_4_layer_call_fn_444595

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
GPU 2J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_443657o
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
�"
�
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_444112

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
�
�
(__inference_model_4_layer_call_fn_443712
input_5
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
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_443657o
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
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_5
�
�
.__inference_quant_dense_7_layer_call_fn_445361

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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_443642o
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
��
� 
C__inference_model_4_layer_call_and_return_conditional_losses_444728

inputs\
Rquantize_layer_4_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: ^
Tquantize_layer_4_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: ^
Lquant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@X
Nquant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: X
Nquant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: ;
-quant_dense_4_biasadd_readvariableop_resource:Y
Oquant_dense_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: [
Qquant_dense_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: ^
Lquant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:X
Nquant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: X
Nquant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: ;
-quant_dense_5_biasadd_readvariableop_resource:Y
Oquant_dense_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: [
Qquant_dense_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: ^
Lquant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:X
Nquant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: X
Nquant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: ;
-quant_dense_6_biasadd_readvariableop_resource:Y
Oquant_dense_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: [
Qquant_dense_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: ^
Lquant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@X
Nquant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: X
Nquant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: ;
-quant_dense_7_biasadd_readvariableop_resource:@Y
Oquant_dense_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: [
Qquant_dense_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��$quant_dense_4/BiasAdd/ReadVariableOp�Cquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Fquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�$quant_dense_5/BiasAdd/ReadVariableOp�Cquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Fquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�$quant_dense_6/BiasAdd/ReadVariableOp�Cquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Fquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�$quant_dense_7/BiasAdd/ReadVariableOp�Cquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Fquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Iquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Kquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Iquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpRquantize_layer_4_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Kquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpTquantize_layer_4_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
:quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsQquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Squantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Cquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpLquant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpNquant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpNquant_dense_4_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
4quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
quant_dense_4/MatMulMatMulDquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
$quant_dense_4/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_4/BiasAddBiasAddquant_dense_4/MatMul:product:0,quant_dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
quant_dense_4/ReluReluquant_dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Fquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_dense_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Hquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_dense_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_dense_4/Relu:activations:0Nquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Cquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpLquant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpNquant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpNquant_dense_5_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
4quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_5/MatMulMatMulAquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
$quant_dense_5/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_5/BiasAddBiasAddquant_dense_5/MatMul:product:0,quant_dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
quant_dense_5/ReluReluquant_dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Fquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_dense_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Hquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_dense_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_dense_5/Relu:activations:0Nquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Cquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpLquant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpNquant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpNquant_dense_6_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
4quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_6/MatMulMatMulAquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
$quant_dense_6/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_6/BiasAddBiasAddquant_dense_6/MatMul:product:0,quant_dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
quant_dense_6/ReluReluquant_dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Fquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_dense_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Hquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_dense_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_dense_6/Relu:activations:0Nquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Cquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpLquant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@*
dtype0�
Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpNquant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpNquant_dense_7_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
4quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@*
narrow_range(�
quant_dense_7/MatMulMatMulAquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
$quant_dense_7/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_7/BiasAddBiasAddquant_dense_7/MatMul:product:0,quant_dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Fquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_dense_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Hquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_dense_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_7/BiasAdd:output:0Nquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityAquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp%^quant_dense_4/BiasAdd/ReadVariableOpD^quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2G^quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_5/BiasAdd/ReadVariableOpD^quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2G^quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_6/BiasAdd/ReadVariableOpD^quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2G^quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_7/BiasAdd/ReadVariableOpD^quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2G^quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1J^quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpL^quantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$quant_dense_4/BiasAdd/ReadVariableOp$quant_dense_4/BiasAdd/ReadVariableOp2�
Cquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_4/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Fquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_5/BiasAdd/ReadVariableOp$quant_dense_5/BiasAdd/ReadVariableOp2�
Cquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_5/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Fquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_6/BiasAdd/ReadVariableOp$quant_dense_6/BiasAdd/ReadVariableOp2�
Cquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_6/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Fquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_7/BiasAdd/ReadVariableOp$quant_dense_7/BiasAdd/ReadVariableOp2�
Cquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_7/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Fquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Iquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpIquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Kquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Kquantize_layer_4/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_6_layer_call_fn_445266

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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_443880o
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
�%
�	
C__inference_model_4_layer_call_and_return_conditional_losses_444412
input_5!
quantize_layer_4_444354: !
quantize_layer_4_444356: &
quant_dense_4_444359:@
quant_dense_4_444361: 
quant_dense_4_444363: "
quant_dense_4_444365:
quant_dense_4_444367: 
quant_dense_4_444369: &
quant_dense_5_444372:
quant_dense_5_444374: 
quant_dense_5_444376: "
quant_dense_5_444378:
quant_dense_5_444380: 
quant_dense_5_444382: &
quant_dense_6_444385:
quant_dense_6_444387: 
quant_dense_6_444389: "
quant_dense_6_444391:
quant_dense_6_444393: 
quant_dense_6_444395: &
quant_dense_7_444398:@
quant_dense_7_444400: 
quant_dense_7_444402: "
quant_dense_7_444404:@
quant_dense_7_444406: 
quant_dense_7_444408: 
identity��%quant_dense_4/StatefulPartitionedCall�%quant_dense_5/StatefulPartitionedCall�%quant_dense_6/StatefulPartitionedCall�%quant_dense_7/StatefulPartitionedCall�(quantize_layer_4/StatefulPartitionedCall�
(quantize_layer_4/StatefulPartitionedCallStatefulPartitionedCallinput_5quantize_layer_4_444354quantize_layer_4_444356*
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
GPU 2J 8� *U
fPRN
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_443511�
%quant_dense_4/StatefulPartitionedCallStatefulPartitionedCall1quantize_layer_4/StatefulPartitionedCall:output:0quant_dense_4_444359quant_dense_4_444361quant_dense_4_444363quant_dense_4_444365quant_dense_4_444367quant_dense_4_444369*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_443538�
%quant_dense_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_4/StatefulPartitionedCall:output:0quant_dense_5_444372quant_dense_5_444374quant_dense_5_444376quant_dense_5_444378quant_dense_5_444380quant_dense_5_444382*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_443573�
%quant_dense_6/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_5/StatefulPartitionedCall:output:0quant_dense_6_444385quant_dense_6_444387quant_dense_6_444389quant_dense_6_444391quant_dense_6_444393quant_dense_6_444395*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_443608�
%quant_dense_7/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_6/StatefulPartitionedCall:output:0quant_dense_7_444398quant_dense_7_444400quant_dense_7_444402quant_dense_7_444404quant_dense_7_444406quant_dense_7_444408*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_443642}
IdentityIdentity.quant_dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp&^quant_dense_4/StatefulPartitionedCall&^quant_dense_5/StatefulPartitionedCall&^quant_dense_6/StatefulPartitionedCall&^quant_dense_7/StatefulPartitionedCall)^quantize_layer_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%quant_dense_4/StatefulPartitionedCall%quant_dense_4/StatefulPartitionedCall2N
%quant_dense_5/StatefulPartitionedCall%quant_dense_5/StatefulPartitionedCall2N
%quant_dense_6/StatefulPartitionedCall%quant_dense_6/StatefulPartitionedCall2N
%quant_dense_7/StatefulPartitionedCall%quant_dense_7/StatefulPartitionedCall2T
(quantize_layer_4/StatefulPartitionedCall(quantize_layer_4/StatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_5
��
�!
"__inference__traced_restore_445811
file_prefix@
6assignvariableop_quantize_layer_4_quantize_layer_4_min: B
8assignvariableop_1_quantize_layer_4_quantize_layer_4_max: <
2assignvariableop_2_quantize_layer_4_optimizer_step: 9
/assignvariableop_3_quant_dense_4_optimizer_step: 5
+assignvariableop_4_quant_dense_4_kernel_min: 5
+assignvariableop_5_quant_dense_4_kernel_max: >
4assignvariableop_6_quant_dense_4_post_activation_min: >
4assignvariableop_7_quant_dense_4_post_activation_max: 9
/assignvariableop_8_quant_dense_5_optimizer_step: 5
+assignvariableop_9_quant_dense_5_kernel_min: 6
,assignvariableop_10_quant_dense_5_kernel_max: ?
5assignvariableop_11_quant_dense_5_post_activation_min: ?
5assignvariableop_12_quant_dense_5_post_activation_max: :
0assignvariableop_13_quant_dense_6_optimizer_step: 6
,assignvariableop_14_quant_dense_6_kernel_min: 6
,assignvariableop_15_quant_dense_6_kernel_max: ?
5assignvariableop_16_quant_dense_6_post_activation_min: ?
5assignvariableop_17_quant_dense_6_post_activation_max: :
0assignvariableop_18_quant_dense_7_optimizer_step: 6
,assignvariableop_19_quant_dense_7_kernel_min: 6
,assignvariableop_20_quant_dense_7_kernel_max: ?
5assignvariableop_21_quant_dense_7_post_activation_min: ?
5assignvariableop_22_quant_dense_7_post_activation_max: '
assignvariableop_23_adam_iter:	 )
assignvariableop_24_adam_beta_1: )
assignvariableop_25_adam_beta_2: (
assignvariableop_26_adam_decay: 0
&assignvariableop_27_adam_learning_rate: 4
"assignvariableop_28_dense_4_kernel:@.
 assignvariableop_29_dense_4_bias:4
"assignvariableop_30_dense_5_kernel:.
 assignvariableop_31_dense_5_bias:4
"assignvariableop_32_dense_6_kernel:.
 assignvariableop_33_dense_6_bias:4
"assignvariableop_34_dense_7_kernel:@.
 assignvariableop_35_dense_7_bias:@#
assignvariableop_36_total: #
assignvariableop_37_count: ;
)assignvariableop_38_adam_dense_4_kernel_m:@5
'assignvariableop_39_adam_dense_4_bias_m:;
)assignvariableop_40_adam_dense_5_kernel_m:5
'assignvariableop_41_adam_dense_5_bias_m:;
)assignvariableop_42_adam_dense_6_kernel_m:5
'assignvariableop_43_adam_dense_6_bias_m:;
)assignvariableop_44_adam_dense_7_kernel_m:@5
'assignvariableop_45_adam_dense_7_bias_m:@;
)assignvariableop_46_adam_dense_4_kernel_v:@5
'assignvariableop_47_adam_dense_4_bias_v:;
)assignvariableop_48_adam_dense_5_kernel_v:5
'assignvariableop_49_adam_dense_5_bias_v:;
)assignvariableop_50_adam_dense_6_kernel_v:5
'assignvariableop_51_adam_dense_6_bias_v:;
)assignvariableop_52_adam_dense_7_kernel_v:@5
'assignvariableop_53_adam_dense_7_bias_v:@
identity_55��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7BDlayer_with_weights-0/quantize_layer_4_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-0/quantize_layer_4_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-3/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-4/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
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
AssignVariableOpAssignVariableOp6assignvariableop_quantize_layer_4_quantize_layer_4_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp8assignvariableop_1_quantize_layer_4_quantize_layer_4_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp2assignvariableop_2_quantize_layer_4_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_quant_dense_4_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp+assignvariableop_4_quant_dense_4_kernel_minIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_quant_dense_4_kernel_maxIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp4assignvariableop_6_quant_dense_4_post_activation_minIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp4assignvariableop_7_quant_dense_4_post_activation_maxIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_quant_dense_5_optimizer_stepIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_quant_dense_5_kernel_minIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp,assignvariableop_10_quant_dense_5_kernel_maxIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp5assignvariableop_11_quant_dense_5_post_activation_minIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp5assignvariableop_12_quant_dense_5_post_activation_maxIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp0assignvariableop_13_quant_dense_6_optimizer_stepIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp,assignvariableop_14_quant_dense_6_kernel_minIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_quant_dense_6_kernel_maxIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_quant_dense_6_post_activation_minIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp5assignvariableop_17_quant_dense_6_post_activation_maxIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp0assignvariableop_18_quant_dense_7_optimizer_stepIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp,assignvariableop_19_quant_dense_7_kernel_minIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_quant_dense_7_kernel_maxIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp5assignvariableop_21_quant_dense_7_post_activation_minIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp5assignvariableop_22_quant_dense_7_post_activation_maxIdentity_22:output:0"/device:CPU:0*
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
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_4_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp assignvariableop_29_dense_4_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_5_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_5_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_6_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_6_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_7_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_7_biasIdentity_35:output:0"/device:CPU:0*
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
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_4_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_4_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_5_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_5_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_6_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_6_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_7_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_7_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_4_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_dense_4_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_5_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_dense_5_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_6_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_dense_6_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_dense_7_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_dense_7_bias_vIdentity_53:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix
�%
�	
C__inference_model_4_layer_call_and_return_conditional_losses_444239

inputs!
quantize_layer_4_444181: !
quantize_layer_4_444183: &
quant_dense_4_444186:@
quant_dense_4_444188: 
quant_dense_4_444190: "
quant_dense_4_444192:
quant_dense_4_444194: 
quant_dense_4_444196: &
quant_dense_5_444199:
quant_dense_5_444201: 
quant_dense_5_444203: "
quant_dense_5_444205:
quant_dense_5_444207: 
quant_dense_5_444209: &
quant_dense_6_444212:
quant_dense_6_444214: 
quant_dense_6_444216: "
quant_dense_6_444218:
quant_dense_6_444220: 
quant_dense_6_444222: &
quant_dense_7_444225:@
quant_dense_7_444227: 
quant_dense_7_444229: "
quant_dense_7_444231:@
quant_dense_7_444233: 
quant_dense_7_444235: 
identity��%quant_dense_4/StatefulPartitionedCall�%quant_dense_5/StatefulPartitionedCall�%quant_dense_6/StatefulPartitionedCall�%quant_dense_7/StatefulPartitionedCall�(quantize_layer_4/StatefulPartitionedCall�
(quantize_layer_4/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_4_444181quantize_layer_4_444183*
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
GPU 2J 8� *U
fPRN
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_444112�
%quant_dense_4/StatefulPartitionedCallStatefulPartitionedCall1quantize_layer_4/StatefulPartitionedCall:output:0quant_dense_4_444186quant_dense_4_444188quant_dense_4_444190quant_dense_4_444192quant_dense_4_444194quant_dense_4_444196*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_444064�
%quant_dense_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_4/StatefulPartitionedCall:output:0quant_dense_5_444199quant_dense_5_444201quant_dense_5_444203quant_dense_5_444205quant_dense_5_444207quant_dense_5_444209*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_443972�
%quant_dense_6/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_5/StatefulPartitionedCall:output:0quant_dense_6_444212quant_dense_6_444214quant_dense_6_444216quant_dense_6_444218quant_dense_6_444220quant_dense_6_444222*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_443880�
%quant_dense_7/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_6/StatefulPartitionedCall:output:0quant_dense_7_444225quant_dense_7_444227quant_dense_7_444229quant_dense_7_444231quant_dense_7_444233quant_dense_7_444235*
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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_443788}
IdentityIdentity.quant_dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp&^quant_dense_4/StatefulPartitionedCall&^quant_dense_5/StatefulPartitionedCall&^quant_dense_6/StatefulPartitionedCall&^quant_dense_7/StatefulPartitionedCall)^quantize_layer_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%quant_dense_4/StatefulPartitionedCall%quant_dense_4/StatefulPartitionedCall2N
%quant_dense_5/StatefulPartitionedCall%quant_dense_5/StatefulPartitionedCall2N
%quant_dense_6/StatefulPartitionedCall%quant_dense_6/StatefulPartitionedCall2N
%quant_dense_7/StatefulPartitionedCall%quant_dense_7/StatefulPartitionedCall2T
(quantize_layer_4/StatefulPartitionedCall(quantize_layer_4/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_4_layer_call_fn_445042

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
GPU 2J 8� *R
fMRK
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_444064o
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
�T
�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_443788

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
�"
�
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_445008

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
�
�
(__inference_model_4_layer_call_fn_444351
input_5
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
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_444239o
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
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_5
�U
�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_443972

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
�
�
1__inference_quantize_layer_4_layer_call_fn_444969

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
GPU 2J 8� *U
fPRN
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_443511o
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
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_445232

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
�
�
$__inference_signature_wrapper_444538
input_5
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� **
f%R#
!__inference__wrapped_model_443495o
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
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_5"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_50
serving_default_input_5:0���������@A
quant_dense_70
StatefulPartitionedCall:0���������@tensorflow/serving/predict:ߺ
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
quantize_layer_4_min
quantize_layer_4_max
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
-:+ 2%quantize_layer_4/quantize_layer_4_min
-:+ 2%quantize_layer_4/quantize_layer_4_max
:
min_var
max_var"
trackable_dict_wrapper
':% 2quantize_layer_4/optimizer_step
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
$:" 2quant_dense_4/optimizer_step
'
d0"
trackable_list_wrapper
 : 2quant_dense_4/kernel_min
 : 2quant_dense_4/kernel_max
 "
trackable_list_wrapper
):' 2!quant_dense_4/post_activation_min
):' 2!quant_dense_4/post_activation_max
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
$:" 2quant_dense_5/optimizer_step
'
n0"
trackable_list_wrapper
 : 2quant_dense_5/kernel_min
 : 2quant_dense_5/kernel_max
 "
trackable_list_wrapper
):' 2!quant_dense_5/post_activation_min
):' 2!quant_dense_5/post_activation_max
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
$:" 2quant_dense_6/optimizer_step
'
x0"
trackable_list_wrapper
 : 2quant_dense_6/kernel_min
 : 2quant_dense_6/kernel_max
 "
trackable_list_wrapper
):' 2!quant_dense_6/post_activation_min
):' 2!quant_dense_6/post_activation_max
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
$:" 2quant_dense_7/optimizer_step
(
�0"
trackable_list_wrapper
 : 2quant_dense_7/kernel_min
 : 2quant_dense_7/kernel_max
 "
trackable_list_wrapper
):' 2!quant_dense_7/post_activation_min
):' 2!quant_dense_7/post_activation_max
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
 :@2dense_4/kernel
:2dense_4/bias
 :2dense_5/kernel
:2dense_5/bias
 :2dense_6/kernel
:2dense_6/bias
 :@2dense_7/kernel
:@2dense_7/bias
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
%:#@2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
%:#2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
%:#2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
%:#@2Adam/dense_7/kernel/m
:@2Adam/dense_7/bias/m
%:#@2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
%:#2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
%:#2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
%:#@2Adam/dense_7/kernel/v
:@2Adam/dense_7/bias/v
�2�
(__inference_model_4_layer_call_fn_443712
(__inference_model_4_layer_call_fn_444595
(__inference_model_4_layer_call_fn_444652
(__inference_model_4_layer_call_fn_444351�
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
C__inference_model_4_layer_call_and_return_conditional_losses_444728
C__inference_model_4_layer_call_and_return_conditional_losses_444960
C__inference_model_4_layer_call_and_return_conditional_losses_444412
C__inference_model_4_layer_call_and_return_conditional_losses_444473�
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
!__inference__wrapped_model_443495input_5"�
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
1__inference_quantize_layer_4_layer_call_fn_444969
1__inference_quantize_layer_4_layer_call_fn_444978�
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
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_444987
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_445008�
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
.__inference_quant_dense_4_layer_call_fn_445025
.__inference_quant_dense_4_layer_call_fn_445042�
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
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_445063
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_445120�
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
.__inference_quant_dense_5_layer_call_fn_445137
.__inference_quant_dense_5_layer_call_fn_445154�
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
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_445175
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_445232�
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
.__inference_quant_dense_6_layer_call_fn_445249
.__inference_quant_dense_6_layer_call_fn_445266�
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
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_445287
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_445344�
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
.__inference_quant_dense_7_layer_call_fn_445361
.__inference_quant_dense_7_layer_call_fn_445378�
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
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_445398
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_445454�
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
$__inference_signature_wrapper_444538input_5"�
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
!__inference__wrapped_model_443495�NOP%&Q()R23S56T?@UBC0�-
&�#
!�
input_5���������@
� "=�:
8
quant_dense_7'�$
quant_dense_7���������@�
C__inference_model_4_layer_call_and_return_conditional_losses_444412}NOP%&Q()R23S56T?@UBC8�5
.�+
!�
input_5���������@
p 

 
� "%�"
�
0���������@
� �
C__inference_model_4_layer_call_and_return_conditional_losses_444473}NOP%&Q()R23S56T?@UBC8�5
.�+
!�
input_5���������@
p

 
� "%�"
�
0���������@
� �
C__inference_model_4_layer_call_and_return_conditional_losses_444728|NOP%&Q()R23S56T?@UBC7�4
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
C__inference_model_4_layer_call_and_return_conditional_losses_444960|NOP%&Q()R23S56T?@UBC7�4
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
(__inference_model_4_layer_call_fn_443712pNOP%&Q()R23S56T?@UBC8�5
.�+
!�
input_5���������@
p 

 
� "����������@�
(__inference_model_4_layer_call_fn_444351pNOP%&Q()R23S56T?@UBC8�5
.�+
!�
input_5���������@
p

 
� "����������@�
(__inference_model_4_layer_call_fn_444595oNOP%&Q()R23S56T?@UBC7�4
-�*
 �
inputs���������@
p 

 
� "����������@�
(__inference_model_4_layer_call_fn_444652oNOP%&Q()R23S56T?@UBC7�4
-�*
 �
inputs���������@
p

 
� "����������@�
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_445063dNO3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������
� �
I__inference_quant_dense_4_layer_call_and_return_conditional_losses_445120dNO3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������
� �
.__inference_quant_dense_4_layer_call_fn_445025WNO3�0
)�&
 �
inputs���������@
p 
� "�����������
.__inference_quant_dense_4_layer_call_fn_445042WNO3�0
)�&
 �
inputs���������@
p
� "�����������
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_445175dP%&Q()3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_445232dP%&Q()3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
.__inference_quant_dense_5_layer_call_fn_445137WP%&Q()3�0
)�&
 �
inputs���������
p 
� "�����������
.__inference_quant_dense_5_layer_call_fn_445154WP%&Q()3�0
)�&
 �
inputs���������
p
� "�����������
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_445287dR23S563�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_445344dR23S563�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
.__inference_quant_dense_6_layer_call_fn_445249WR23S563�0
)�&
 �
inputs���������
p 
� "�����������
.__inference_quant_dense_6_layer_call_fn_445266WR23S563�0
)�&
 �
inputs���������
p
� "�����������
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_445398dT?@UBC3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������@
� �
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_445454dT?@UBC3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������@
� �
.__inference_quant_dense_7_layer_call_fn_445361WT?@UBC3�0
)�&
 �
inputs���������
p 
� "����������@�
.__inference_quant_dense_7_layer_call_fn_445378WT?@UBC3�0
)�&
 �
inputs���������
p
� "����������@�
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_444987`3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
L__inference_quantize_layer_4_layer_call_and_return_conditional_losses_445008`3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
1__inference_quantize_layer_4_layer_call_fn_444969S3�0
)�&
 �
inputs���������@
p 
� "����������@�
1__inference_quantize_layer_4_layer_call_fn_444978S3�0
)�&
 �
inputs���������@
p
� "����������@�
$__inference_signature_wrapper_444538�NOP%&Q()R23S56T?@UBC;�8
� 
1�.
,
input_5!�
input_5���������@"=�:
8
quant_dense_7'�$
quant_dense_7���������@