��
��
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
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
�
'quantize_layer_11/quantize_layer_11_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'quantize_layer_11/quantize_layer_11_min
�
;quantize_layer_11/quantize_layer_11_min/Read/ReadVariableOpReadVariableOp'quantize_layer_11/quantize_layer_11_min*
_output_shapes
: *
dtype0
�
'quantize_layer_11/quantize_layer_11_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'quantize_layer_11/quantize_layer_11_max
�
;quantize_layer_11/quantize_layer_11_max/Read/ReadVariableOpReadVariableOp'quantize_layer_11/quantize_layer_11_max*
_output_shapes
: *
dtype0
�
 quantize_layer_11/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quantize_layer_11/optimizer_step
�
4quantize_layer_11/optimizer_step/Read/ReadVariableOpReadVariableOp quantize_layer_11/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_35/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_dense_35/optimizer_step
�
1quant_dense_35/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_35/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_35/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_35/kernel_min

-quant_dense_35/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_35/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_35/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namequant_dense_35/kernel_max

-quant_dense_35/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_35/kernel_max*
_output_shapes
: *
dtype0
�
"quant_dense_35/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_35/post_activation_min
�
6quant_dense_35/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_dense_35/post_activation_min*
_output_shapes
: *
dtype0
�
"quant_dense_35/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_dense_35/post_activation_max
�
6quant_dense_35/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_dense_35/post_activation_max*
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
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:@@*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
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
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_35/kernel/m
�
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

:@@*
dtype0
�
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_35/kernel/v
�
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

:@@*
dtype0
�
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
�!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�!
value�!B�! B�!
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
 
�

quantize_layer_11_min
quantize_layer_11_max
quantizer_vars
optimizer_step
	variables
trainable_variables
regularization_losses
	keras_api
�
	layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers
	variables
trainable_variables
regularization_losses
	keras_api
d
iter

 beta_1

!beta_2
	"decay
#learning_rate$mE%mF$vG%vH
F

0
1
2
$3
%4
5
6
7
8
9

$0
%1
 
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
 
��
VARIABLE_VALUE'quantize_layer_11/quantize_layer_11_minElayer_with_weights-0/quantize_layer_11_min/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'quantize_layer_11/quantize_layer_11_maxElayer_with_weights-0/quantize_layer_11_max/.ATTRIBUTES/VARIABLE_VALUE


min_var
max_var
tr
VARIABLE_VALUE quantize_layer_11/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE


0
1
2
 
 
�
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
h

$kernel
%bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
qo
VARIABLE_VALUEquant_dense_35/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

40
ig
VARIABLE_VALUEquant_dense_35/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_dense_35/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_dense_35/post_activation_minClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_dense_35/post_activation_maxClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
1
$0
%1
2
3
4
5
6

$0
%1
 
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
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
VARIABLE_VALUEdense_35/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_35/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
8

0
1
2
3
4
5
6
7

0
1
2

:0
 
 


0
1
2
 
 
 
 

%0

%0
 
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses

$0
@2
#
0
1
2
3
4

0
 
 
 
4
	Atotal
	Bcount
C	variables
D	keras_api
 
 
 
 
 

min_var
max_var
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

C	variables
nl
VARIABLE_VALUEAdam/dense_35/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_35/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_35/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_35/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_12Placeholder*'
_output_shapes
:���������@*
dtype0*
shape:���������@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_12'quantize_layer_11/quantize_layer_11_min'quantize_layer_11/quantize_layer_11_maxdense_35/kernelquant_dense_35/kernel_minquant_dense_35/kernel_maxdense_35/bias"quant_dense_35/post_activation_min"quant_dense_35/post_activation_max*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_222717
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename;quantize_layer_11/quantize_layer_11_min/Read/ReadVariableOp;quantize_layer_11/quantize_layer_11_max/Read/ReadVariableOp4quantize_layer_11/optimizer_step/Read/ReadVariableOp1quant_dense_35/optimizer_step/Read/ReadVariableOp-quant_dense_35/kernel_min/Read/ReadVariableOp-quant_dense_35/kernel_max/Read/ReadVariableOp6quant_dense_35/post_activation_min/Read/ReadVariableOp6quant_dense_35/post_activation_max/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
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
__inference__traced_save_223101
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename'quantize_layer_11/quantize_layer_11_min'quantize_layer_11/quantize_layer_11_max quantize_layer_11/optimizer_stepquant_dense_35/optimizer_stepquant_dense_35/kernel_minquant_dense_35/kernel_max"quant_dense_35/post_activation_min"quant_dense_35/post_activation_max	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_35/kerneldense_35/biastotalcountAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/dense_35/kernel/vAdam/dense_35/bias/v*!
Tin
2*
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
"__inference__traced_restore_223174��
�
�
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222959

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
2__inference_quantize_layer_11_layer_call_fn_222866

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
GPU 2J 8� *V
fQRO
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222368o
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
�
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222368

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
�
�
D__inference_model_11_layer_call_and_return_conditional_losses_222688
input_12"
quantize_layer_11_222669: "
quantize_layer_11_222671: '
quant_dense_35_222674:@@
quant_dense_35_222676: 
quant_dense_35_222678: #
quant_dense_35_222680:@
quant_dense_35_222682: 
quant_dense_35_222684: 
identity��&quant_dense_35/StatefulPartitionedCall�)quantize_layer_11/StatefulPartitionedCall�
)quantize_layer_11/StatefulPartitionedCallStatefulPartitionedCallinput_12quantize_layer_11_222669quantize_layer_11_222671*
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
GPU 2J 8� *V
fQRO
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222552�
&quant_dense_35/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_11/StatefulPartitionedCall:output:0quant_dense_35_222674quant_dense_35_222676quant_dense_35_222678quant_dense_35_222680quant_dense_35_222682quant_dense_35_222684*
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
GPU 2J 8� *S
fNRL
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222504~
IdentityIdentity/quant_dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_35/StatefulPartitionedCall*^quantize_layer_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 2P
&quant_dense_35/StatefulPartitionedCall&quant_dense_35/StatefulPartitionedCall2V
)quantize_layer_11/StatefulPartitionedCall)quantize_layer_11/StatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_12
�U
�
"__inference__traced_restore_223174
file_prefixB
8assignvariableop_quantize_layer_11_quantize_layer_11_min: D
:assignvariableop_1_quantize_layer_11_quantize_layer_11_max: =
3assignvariableop_2_quantize_layer_11_optimizer_step: :
0assignvariableop_3_quant_dense_35_optimizer_step: 6
,assignvariableop_4_quant_dense_35_kernel_min: 6
,assignvariableop_5_quant_dense_35_kernel_max: ?
5assignvariableop_6_quant_dense_35_post_activation_min: ?
5assignvariableop_7_quant_dense_35_post_activation_max: &
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: 5
#assignvariableop_13_dense_35_kernel:@@/
!assignvariableop_14_dense_35_bias:@#
assignvariableop_15_total: #
assignvariableop_16_count: <
*assignvariableop_17_adam_dense_35_kernel_m:@@6
(assignvariableop_18_adam_dense_35_bias_m:@<
*assignvariableop_19_adam_dense_35_kernel_v:@@6
(assignvariableop_20_adam_dense_35_bias_v:@
identity_22��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	BElayer_with_weights-0/quantize_layer_11_min/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/quantize_layer_11_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp8assignvariableop_quantize_layer_11_quantize_layer_11_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp:assignvariableop_1_quantize_layer_11_quantize_layer_11_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp3assignvariableop_2_quantize_layer_11_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_quant_dense_35_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_quant_dense_35_kernel_minIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp,assignvariableop_5_quant_dense_35_kernel_maxIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp5assignvariableop_6_quant_dense_35_post_activation_minIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp5assignvariableop_7_quant_dense_35_post_activation_maxIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_35_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_35_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_35_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_35_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_35_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_35_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222884

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
2__inference_quantize_layer_11_layer_call_fn_222875

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
GPU 2J 8� *V
fQRO
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222552o
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
�8
�
!__inference__wrapped_model_222352
input_12f
\model_11_quantize_layer_11_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: h
^model_11_quantize_layer_11_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: h
Vmodel_11_quant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@@b
Xmodel_11_quant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: b
Xmodel_11_quant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: E
7model_11_quant_dense_35_biasadd_readvariableop_resource:@c
Ymodel_11_quant_dense_35_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: e
[model_11_quant_dense_35_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��.model_11/quant_dense_35/BiasAdd/ReadVariableOp�Mmodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Omodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Omodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Pmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Rmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Smodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Umodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Smodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp\model_11_quantize_layer_11_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Umodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp^model_11_quantize_layer_11_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Dmodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinput_12[model_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0]model_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Mmodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpVmodel_11_quant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Omodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpXmodel_11_quant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Omodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpXmodel_11_quant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
>model_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsUmodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Wmodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Wmodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@@*
narrow_range(�
model_11/quant_dense_35/MatMulMatMulNmodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Hmodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
.model_11/quant_dense_35/BiasAdd/ReadVariableOpReadVariableOp7model_11_quant_dense_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_11/quant_dense_35/BiasAddBiasAdd(model_11/quant_dense_35/MatMul:product:06model_11/quant_dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Pmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYmodel_11_quant_dense_35_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Rmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[model_11_quant_dense_35_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Amodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars(model_11/quant_dense_35/BiasAdd:output:0Xmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityKmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp/^model_11/quant_dense_35/BiasAdd/ReadVariableOpN^model_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpP^model_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1P^model_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Q^model_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpS^model_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1T^model_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpV^model_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 2`
.model_11/quant_dense_35/BiasAdd/ReadVariableOp.model_11/quant_dense_35/BiasAdd/ReadVariableOp2�
Mmodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpMmodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Omodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Omodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Omodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Omodel_11/quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Pmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpPmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Rmodel_11/quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Smodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpSmodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Umodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Umodel_11/quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_12
�	
�
)__inference_model_11_layer_call_fn_222644
input_12
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_11_layer_call_and_return_conditional_losses_222604o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_12
�
�
$__inference_signature_wrapper_222717
input_12
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_222352o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_12
�
�
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222394

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
�"
�
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222552

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
�T
�
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_223015

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
�3
�

D__inference_model_11_layer_call_and_return_conditional_losses_222784

inputs]
Squantize_layer_11_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: _
Uquantize_layer_11_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Mquant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:@@Y
Oquant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: Y
Oquant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: <
.quant_dense_35_biasadd_readvariableop_resource:@Z
Pquant_dense_35_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_dense_35_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��%quant_dense_35/BiasAdd/ReadVariableOp�Dquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Gquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Jquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Lquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Jquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpSquantize_layer_11_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Lquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpUquantize_layer_11_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
;quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsRquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Tquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
Dquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpOquant_dense_35_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
5quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@@*
narrow_range(�
quant_dense_35/MatMulMatMulEquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
%quant_dense_35/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_35/BiasAddBiasAddquant_dense_35/MatMul:product:0-quant_dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Gquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_dense_35_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Iquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_dense_35_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_35/BiasAdd:output:0Oquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityBquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp&^quant_dense_35/BiasAdd/ReadVariableOpE^quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2H^quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1K^quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpM^quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 2N
%quant_dense_35/BiasAdd/ReadVariableOp%quant_dense_35/BiasAdd/ReadVariableOp2�
Dquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Gquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Jquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Lquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Lquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�1
�	
__inference__traced_save_223101
file_prefixF
Bsavev2_quantize_layer_11_quantize_layer_11_min_read_readvariableopF
Bsavev2_quantize_layer_11_quantize_layer_11_max_read_readvariableop?
;savev2_quantize_layer_11_optimizer_step_read_readvariableop<
8savev2_quant_dense_35_optimizer_step_read_readvariableop8
4savev2_quant_dense_35_kernel_min_read_readvariableop8
4savev2_quant_dense_35_kernel_max_read_readvariableopA
=savev2_quant_dense_35_post_activation_min_read_readvariableopA
=savev2_quant_dense_35_post_activation_max_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop
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
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	BElayer_with_weights-0/quantize_layer_11_min/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/quantize_layer_11_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Bsavev2_quantize_layer_11_quantize_layer_11_min_read_readvariableopBsavev2_quantize_layer_11_quantize_layer_11_max_read_readvariableop;savev2_quantize_layer_11_optimizer_step_read_readvariableop8savev2_quant_dense_35_optimizer_step_read_readvariableop4savev2_quant_dense_35_kernel_min_read_readvariableop4savev2_quant_dense_35_kernel_max_read_readvariableop=savev2_quant_dense_35_post_activation_min_read_readvariableop=savev2_quant_dense_35_post_activation_max_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	�
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

identity_1Identity_1:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : :@@:@: : :@@:@:@@:@: 2(
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
: :$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: 
�	
�
)__inference_model_11_layer_call_fn_222759

inputs
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_11_layer_call_and_return_conditional_losses_222604o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
)__inference_model_11_layer_call_fn_222428
input_12
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_11_layer_call_and_return_conditional_losses_222409o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_12
�
�
D__inference_model_11_layer_call_and_return_conditional_losses_222604

inputs"
quantize_layer_11_222585: "
quantize_layer_11_222587: '
quant_dense_35_222590:@@
quant_dense_35_222592: 
quant_dense_35_222594: #
quant_dense_35_222596:@
quant_dense_35_222598: 
quant_dense_35_222600: 
identity��&quant_dense_35/StatefulPartitionedCall�)quantize_layer_11/StatefulPartitionedCall�
)quantize_layer_11/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_11_222585quantize_layer_11_222587*
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
GPU 2J 8� *V
fQRO
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222552�
&quant_dense_35/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_11/StatefulPartitionedCall:output:0quant_dense_35_222590quant_dense_35_222592quant_dense_35_222594quant_dense_35_222596quant_dense_35_222598quant_dense_35_222600*
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
GPU 2J 8� *S
fNRL
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222504~
IdentityIdentity/quant_dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_35/StatefulPartitionedCall*^quantize_layer_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 2P
&quant_dense_35/StatefulPartitionedCall&quant_dense_35/StatefulPartitionedCall2V
)quantize_layer_11/StatefulPartitionedCall)quantize_layer_11/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
Β
�
D__inference_model_11_layer_call_and_return_conditional_losses_222857

inputsM
Cquantize_layer_11_allvaluesquantize_minimum_readvariableop_resource: M
Cquantize_layer_11_allvaluesquantize_maximum_readvariableop_resource: L
:quant_dense_35_lastvaluequant_rank_readvariableop_resource:@@>
4quant_dense_35_lastvaluequant_assignminlast_resource: >
4quant_dense_35_lastvaluequant_assignmaxlast_resource: <
.quant_dense_35_biasadd_readvariableop_resource:@O
Equant_dense_35_movingavgquantize_assignminema_readvariableop_resource: O
Equant_dense_35_movingavgquantize_assignmaxema_readvariableop_resource: 
identity��%quant_dense_35/BiasAdd/ReadVariableOp�+quant_dense_35/LastValueQuant/AssignMaxLast�+quant_dense_35/LastValueQuant/AssignMinLast�5quant_dense_35/LastValueQuant/BatchMax/ReadVariableOp�5quant_dense_35/LastValueQuant/BatchMin/ReadVariableOp�Dquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Aquant_dense_35/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�<quant_dense_35/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Aquant_dense_35/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�<quant_dense_35/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Gquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Iquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�5quantize_layer_11/AllValuesQuantize/AssignMaxAllValue�5quantize_layer_11/AllValuesQuantize/AssignMinAllValue�Jquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Lquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�:quantize_layer_11/AllValuesQuantize/Maximum/ReadVariableOp�:quantize_layer_11/AllValuesQuantize/Minimum/ReadVariableOpz
)quantize_layer_11/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
,quantize_layer_11/AllValuesQuantize/BatchMinMininputs2quantize_layer_11/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: |
+quantize_layer_11/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
,quantize_layer_11/AllValuesQuantize/BatchMaxMaxinputs4quantize_layer_11/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
:quantize_layer_11/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOpCquantize_layer_11_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
+quantize_layer_11/AllValuesQuantize/MinimumMinimumBquantize_layer_11/AllValuesQuantize/Minimum/ReadVariableOp:value:05quantize_layer_11/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: t
/quantize_layer_11/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-quantize_layer_11/AllValuesQuantize/Minimum_1Minimum/quantize_layer_11/AllValuesQuantize/Minimum:z:08quantize_layer_11/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
:quantize_layer_11/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOpCquantize_layer_11_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
+quantize_layer_11/AllValuesQuantize/MaximumMaximumBquantize_layer_11/AllValuesQuantize/Maximum/ReadVariableOp:value:05quantize_layer_11/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: t
/quantize_layer_11/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-quantize_layer_11/AllValuesQuantize/Maximum_1Maximum/quantize_layer_11/AllValuesQuantize/Maximum:z:08quantize_layer_11/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
5quantize_layer_11/AllValuesQuantize/AssignMinAllValueAssignVariableOpCquantize_layer_11_allvaluesquantize_minimum_readvariableop_resource1quantize_layer_11/AllValuesQuantize/Minimum_1:z:0;^quantize_layer_11/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0�
5quantize_layer_11/AllValuesQuantize/AssignMaxAllValueAssignVariableOpCquantize_layer_11_allvaluesquantize_maximum_readvariableop_resource1quantize_layer_11/AllValuesQuantize/Maximum_1:z:0;^quantize_layer_11/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0�
Jquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpCquantize_layer_11_allvaluesquantize_minimum_readvariableop_resource6^quantize_layer_11/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
Lquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCquantize_layer_11_allvaluesquantize_maximum_readvariableop_resource6^quantize_layer_11/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
;quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsRquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Tquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
1quant_dense_35/LastValueQuant/Rank/ReadVariableOpReadVariableOp:quant_dense_35_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0d
"quant_dense_35/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)quant_dense_35/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)quant_dense_35/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#quant_dense_35/LastValueQuant/rangeRange2quant_dense_35/LastValueQuant/range/start:output:0+quant_dense_35/LastValueQuant/Rank:output:02quant_dense_35/LastValueQuant/range/delta:output:0*
_output_shapes
:�
5quant_dense_35/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp:quant_dense_35_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0�
&quant_dense_35/LastValueQuant/BatchMinMin=quant_dense_35/LastValueQuant/BatchMin/ReadVariableOp:value:0,quant_dense_35/LastValueQuant/range:output:0*
T0*
_output_shapes
: �
3quant_dense_35/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp:quant_dense_35_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0f
$quant_dense_35/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :m
+quant_dense_35/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+quant_dense_35/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
%quant_dense_35/LastValueQuant/range_1Range4quant_dense_35/LastValueQuant/range_1/start:output:0-quant_dense_35/LastValueQuant/Rank_1:output:04quant_dense_35/LastValueQuant/range_1/delta:output:0*
_output_shapes
:�
5quant_dense_35/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp:quant_dense_35_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0�
&quant_dense_35/LastValueQuant/BatchMaxMax=quant_dense_35/LastValueQuant/BatchMax/ReadVariableOp:value:0.quant_dense_35/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: l
'quant_dense_35/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
%quant_dense_35/LastValueQuant/truedivRealDiv/quant_dense_35/LastValueQuant/BatchMax:output:00quant_dense_35/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
%quant_dense_35/LastValueQuant/MinimumMinimum/quant_dense_35/LastValueQuant/BatchMin:output:0)quant_dense_35/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: h
#quant_dense_35/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
!quant_dense_35/LastValueQuant/mulMul/quant_dense_35/LastValueQuant/BatchMin:output:0,quant_dense_35/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
%quant_dense_35/LastValueQuant/MaximumMaximum/quant_dense_35/LastValueQuant/BatchMax:output:0%quant_dense_35/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
+quant_dense_35/LastValueQuant/AssignMinLastAssignVariableOp4quant_dense_35_lastvaluequant_assignminlast_resource)quant_dense_35/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0�
+quant_dense_35/LastValueQuant/AssignMaxLastAssignVariableOp4quant_dense_35_lastvaluequant_assignmaxlast_resource)quant_dense_35/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0�
Dquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp:quant_dense_35_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp4quant_dense_35_lastvaluequant_assignminlast_resource,^quant_dense_35/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp4quant_dense_35_lastvaluequant_assignmaxlast_resource,^quant_dense_35/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
5quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsLquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Nquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:@@*
narrow_range(�
quant_dense_35/MatMulMatMulEquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0?quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������@�
%quant_dense_35/BiasAdd/ReadVariableOpReadVariableOp.quant_dense_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
quant_dense_35/BiasAddBiasAddquant_dense_35/MatMul:product:0-quant_dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@w
&quant_dense_35/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_35/MovingAvgQuantize/BatchMinMinquant_dense_35/BiasAdd:output:0/quant_dense_35/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: y
(quant_dense_35/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
)quant_dense_35/MovingAvgQuantize/BatchMaxMaxquant_dense_35/BiasAdd:output:01quant_dense_35/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: o
*quant_dense_35/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_35/MovingAvgQuantize/MinimumMinimum2quant_dense_35/MovingAvgQuantize/BatchMin:output:03quant_dense_35/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: o
*quant_dense_35/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(quant_dense_35/MovingAvgQuantize/MaximumMaximum2quant_dense_35/MovingAvgQuantize/BatchMax:output:03quant_dense_35/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: x
3quant_dense_35/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_35/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_dense_35_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_35/MovingAvgQuantize/AssignMinEma/subSubDquant_dense_35/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_dense_35/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
1quant_dense_35/MovingAvgQuantize/AssignMinEma/mulMul5quant_dense_35/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_dense_35/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_35/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_35_movingavgquantize_assignminema_readvariableop_resource5quant_dense_35/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_dense_35/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0x
3quant_dense_35/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
<quant_dense_35/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_dense_35_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
1quant_dense_35/MovingAvgQuantize/AssignMaxEma/subSubDquant_dense_35/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_dense_35/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
1quant_dense_35/MovingAvgQuantize/AssignMaxEma/mulMul5quant_dense_35/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_dense_35/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Aquant_dense_35/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_dense_35_movingavgquantize_assignmaxema_readvariableop_resource5quant_dense_35/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_dense_35/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Gquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_dense_35_movingavgquantize_assignminema_readvariableop_resourceB^quant_dense_35/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Iquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_dense_35_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_dense_35/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
8quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_35/BiasAdd:output:0Oquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������@�
IdentityIdentityBquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:���������@�

NoOpNoOp&^quant_dense_35/BiasAdd/ReadVariableOp,^quant_dense_35/LastValueQuant/AssignMaxLast,^quant_dense_35/LastValueQuant/AssignMinLast6^quant_dense_35/LastValueQuant/BatchMax/ReadVariableOp6^quant_dense_35/LastValueQuant/BatchMin/ReadVariableOpE^quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1G^quant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2B^quant_dense_35/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_dense_35/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_dense_35/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_dense_35/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_16^quantize_layer_11/AllValuesQuantize/AssignMaxAllValue6^quantize_layer_11/AllValuesQuantize/AssignMinAllValueK^quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpM^quantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1;^quantize_layer_11/AllValuesQuantize/Maximum/ReadVariableOp;^quantize_layer_11/AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 2N
%quant_dense_35/BiasAdd/ReadVariableOp%quant_dense_35/BiasAdd/ReadVariableOp2Z
+quant_dense_35/LastValueQuant/AssignMaxLast+quant_dense_35/LastValueQuant/AssignMaxLast2Z
+quant_dense_35/LastValueQuant/AssignMinLast+quant_dense_35/LastValueQuant/AssignMinLast2n
5quant_dense_35/LastValueQuant/BatchMax/ReadVariableOp5quant_dense_35/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_dense_35/LastValueQuant/BatchMin/ReadVariableOp5quant_dense_35/LastValueQuant/BatchMin/ReadVariableOp2�
Dquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Fquant_dense_35/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Aquant_dense_35/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_dense_35/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_dense_35/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_dense_35/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Aquant_dense_35/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_dense_35/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_dense_35/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_dense_35/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Gquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Iquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_dense_35/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12n
5quantize_layer_11/AllValuesQuantize/AssignMaxAllValue5quantize_layer_11/AllValuesQuantize/AssignMaxAllValue2n
5quantize_layer_11/AllValuesQuantize/AssignMinAllValue5quantize_layer_11/AllValuesQuantize/AssignMinAllValue2�
Jquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Lquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Lquantize_layer_11/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12x
:quantize_layer_11/AllValuesQuantize/Maximum/ReadVariableOp:quantize_layer_11/AllValuesQuantize/Maximum/ReadVariableOp2x
:quantize_layer_11/AllValuesQuantize/Minimum/ReadVariableOp:quantize_layer_11/AllValuesQuantize/Minimum/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
)__inference_model_11_layer_call_fn_222738

inputs
unknown: 
	unknown_0: 
	unknown_1:@@
	unknown_2: 
	unknown_3: 
	unknown_4:@
	unknown_5: 
	unknown_6: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_11_layer_call_and_return_conditional_losses_222409o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�"
�
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222905

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
�T
�
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222504

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
�
�
D__inference_model_11_layer_call_and_return_conditional_losses_222409

inputs"
quantize_layer_11_222369: "
quantize_layer_11_222371: '
quant_dense_35_222395:@@
quant_dense_35_222397: 
quant_dense_35_222399: #
quant_dense_35_222401:@
quant_dense_35_222403: 
quant_dense_35_222405: 
identity��&quant_dense_35/StatefulPartitionedCall�)quantize_layer_11/StatefulPartitionedCall�
)quantize_layer_11/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_11_222369quantize_layer_11_222371*
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
GPU 2J 8� *V
fQRO
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222368�
&quant_dense_35/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_11/StatefulPartitionedCall:output:0quant_dense_35_222395quant_dense_35_222397quant_dense_35_222399quant_dense_35_222401quant_dense_35_222403quant_dense_35_222405*
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
GPU 2J 8� *S
fNRL
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222394~
IdentityIdentity/quant_dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_35/StatefulPartitionedCall*^quantize_layer_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 2P
&quant_dense_35/StatefulPartitionedCall&quant_dense_35/StatefulPartitionedCall2V
)quantize_layer_11/StatefulPartitionedCall)quantize_layer_11/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
/__inference_quant_dense_35_layer_call_fn_222939

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
GPU 2J 8� *S
fNRL
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222504o
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
�
�
/__inference_quant_dense_35_layer_call_fn_222922

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
GPU 2J 8� *S
fNRL
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222394o
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
�
�
D__inference_model_11_layer_call_and_return_conditional_losses_222666
input_12"
quantize_layer_11_222647: "
quantize_layer_11_222649: '
quant_dense_35_222652:@@
quant_dense_35_222654: 
quant_dense_35_222656: #
quant_dense_35_222658:@
quant_dense_35_222660: 
quant_dense_35_222662: 
identity��&quant_dense_35/StatefulPartitionedCall�)quantize_layer_11/StatefulPartitionedCall�
)quantize_layer_11/StatefulPartitionedCallStatefulPartitionedCallinput_12quantize_layer_11_222647quantize_layer_11_222649*
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
GPU 2J 8� *V
fQRO
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222368�
&quant_dense_35/StatefulPartitionedCallStatefulPartitionedCall2quantize_layer_11/StatefulPartitionedCall:output:0quant_dense_35_222652quant_dense_35_222654quant_dense_35_222656quant_dense_35_222658quant_dense_35_222660quant_dense_35_222662*
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
GPU 2J 8� *S
fNRL
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222394~
IdentityIdentity/quant_dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp'^quant_dense_35/StatefulPartitionedCall*^quantize_layer_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : : : : : 2P
&quant_dense_35/StatefulPartitionedCall&quant_dense_35/StatefulPartitionedCall2V
)quantize_layer_11/StatefulPartitionedCall)quantize_layer_11/StatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
input_12"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_121
serving_default_input_12:0���������@B
quant_dense_350
StatefulPartitionedCall:0���������@tensorflow/serving/predict:�U
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
I__call__
*J&call_and_return_all_conditional_losses
K_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�

quantize_layer_11_min
quantize_layer_11_max
quantizer_vars
optimizer_step
	variables
trainable_variables
regularization_losses
	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers
	variables
trainable_variables
regularization_losses
	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
w
iter

 beta_1

!beta_2
	"decay
#learning_rate$mE%mF$vG%vH"
	optimizer
f

0
1
2
$3
%4
5
6
7
8
9"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
I__call__
K_default_save_signature
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
,
Pserving_default"
signature_map
/:- 2'quantize_layer_11/quantize_layer_11_min
/:- 2'quantize_layer_11/quantize_layer_11_max
:

min_var
max_var"
trackable_dict_wrapper
(:& 2 quantize_layer_11/optimizer_step
5

0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�

$kernel
%bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_dense_35/optimizer_step
'
40"
trackable_list_wrapper
!: 2quant_dense_35/kernel_min
!: 2quant_dense_35/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_dense_35/post_activation_min
*:( 2"quant_dense_35/post_activation_max
 "
trackable_list_wrapper
Q
$0
%1
2
3
4
5
6"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
!:@@2dense_35/kernel
:@2dense_35/bias
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5

0
1
2"
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
%0"
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
/
$0
@2"
trackable_tuple_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Atotal
	Bcount
C	variables
D	keras_api"
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
min_var
max_var"
trackable_dict_wrapper
:  (2total
:  (2count
.
A0
B1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
&:$@@2Adam/dense_35/kernel/m
 :@2Adam/dense_35/bias/m
&:$@@2Adam/dense_35/kernel/v
 :@2Adam/dense_35/bias/v
�2�
)__inference_model_11_layer_call_fn_222428
)__inference_model_11_layer_call_fn_222738
)__inference_model_11_layer_call_fn_222759
)__inference_model_11_layer_call_fn_222644�
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
D__inference_model_11_layer_call_and_return_conditional_losses_222784
D__inference_model_11_layer_call_and_return_conditional_losses_222857
D__inference_model_11_layer_call_and_return_conditional_losses_222666
D__inference_model_11_layer_call_and_return_conditional_losses_222688�
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
!__inference__wrapped_model_222352input_12"�
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
2__inference_quantize_layer_11_layer_call_fn_222866
2__inference_quantize_layer_11_layer_call_fn_222875�
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
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222884
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222905�
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
/__inference_quant_dense_35_layer_call_fn_222922
/__inference_quant_dense_35_layer_call_fn_222939�
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
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222959
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_223015�
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
$__inference_signature_wrapper_222717input_12"�
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
 �
!__inference__wrapped_model_222352~
$%1�.
'�$
"�
input_12���������@
� "?�<
:
quant_dense_35(�%
quant_dense_35���������@�
D__inference_model_11_layer_call_and_return_conditional_losses_222666l
$%9�6
/�,
"�
input_12���������@
p 

 
� "%�"
�
0���������@
� �
D__inference_model_11_layer_call_and_return_conditional_losses_222688l
$%9�6
/�,
"�
input_12���������@
p

 
� "%�"
�
0���������@
� �
D__inference_model_11_layer_call_and_return_conditional_losses_222784j
$%7�4
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
D__inference_model_11_layer_call_and_return_conditional_losses_222857j
$%7�4
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
)__inference_model_11_layer_call_fn_222428_
$%9�6
/�,
"�
input_12���������@
p 

 
� "����������@�
)__inference_model_11_layer_call_fn_222644_
$%9�6
/�,
"�
input_12���������@
p

 
� "����������@�
)__inference_model_11_layer_call_fn_222738]
$%7�4
-�*
 �
inputs���������@
p 

 
� "����������@�
)__inference_model_11_layer_call_fn_222759]
$%7�4
-�*
 �
inputs���������@
p

 
� "����������@�
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_222959d$%3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
J__inference_quant_dense_35_layer_call_and_return_conditional_losses_223015d$%3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
/__inference_quant_dense_35_layer_call_fn_222922W$%3�0
)�&
 �
inputs���������@
p 
� "����������@�
/__inference_quant_dense_35_layer_call_fn_222939W$%3�0
)�&
 �
inputs���������@
p
� "����������@�
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222884`
3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
M__inference_quantize_layer_11_layer_call_and_return_conditional_losses_222905`
3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
2__inference_quantize_layer_11_layer_call_fn_222866S
3�0
)�&
 �
inputs���������@
p 
� "����������@�
2__inference_quantize_layer_11_layer_call_fn_222875S
3�0
)�&
 �
inputs���������@
p
� "����������@�
$__inference_signature_wrapper_222717�
$%=�:
� 
3�0
.
input_12"�
input_12���������@"?�<
:
quant_dense_35(�%
quant_dense_35���������@