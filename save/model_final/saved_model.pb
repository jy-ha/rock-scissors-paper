??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??

?
rsp__model/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namersp__model/conv2d/kernel
?
,rsp__model/conv2d/kernel/Read/ReadVariableOpReadVariableOprsp__model/conv2d/kernel*&
_output_shapes
:@*
dtype0
?
rsp__model/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namersp__model/conv2d/bias
}
*rsp__model/conv2d/bias/Read/ReadVariableOpReadVariableOprsp__model/conv2d/bias*
_output_shapes
:@*
dtype0
?
rsp__model/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_namersp__model/conv2d_1/kernel
?
.rsp__model/conv2d_1/kernel/Read/ReadVariableOpReadVariableOprsp__model/conv2d_1/kernel*&
_output_shapes
:@@*
dtype0
?
rsp__model/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namersp__model/conv2d_1/bias
?
,rsp__model/conv2d_1/bias/Read/ReadVariableOpReadVariableOprsp__model/conv2d_1/bias*
_output_shapes
:@*
dtype0
?
rsp__model/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_namersp__model/conv2d_2/kernel
?
.rsp__model/conv2d_2/kernel/Read/ReadVariableOpReadVariableOprsp__model/conv2d_2/kernel*'
_output_shapes
:@?*
dtype0
?
rsp__model/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namersp__model/conv2d_2/bias
?
,rsp__model/conv2d_2/bias/Read/ReadVariableOpReadVariableOprsp__model/conv2d_2/bias*
_output_shapes	
:?*
dtype0
?
rsp__model/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_namersp__model/conv2d_3/kernel
?
.rsp__model/conv2d_3/kernel/Read/ReadVariableOpReadVariableOprsp__model/conv2d_3/kernel*(
_output_shapes
:??*
dtype0
?
rsp__model/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namersp__model/conv2d_3/bias
?
,rsp__model/conv2d_3/bias/Read/ReadVariableOpReadVariableOprsp__model/conv2d_3/bias*
_output_shapes	
:?*
dtype0
?
rsp__model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_namersp__model/dense/kernel
?
+rsp__model/dense/kernel/Read/ReadVariableOpReadVariableOprsp__model/dense/kernel* 
_output_shapes
:
??*
dtype0
?
rsp__model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_namersp__model/dense/bias
|
)rsp__model/dense/bias/Read/ReadVariableOpReadVariableOprsp__model/dense/bias*
_output_shapes	
:?*
dtype0
?
rsp__model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namersp__model/dense_1/kernel
?
-rsp__model/dense_1/kernel/Read/ReadVariableOpReadVariableOprsp__model/dense_1/kernel*
_output_shapes
:	?*
dtype0
?
rsp__model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namersp__model/dense_1/bias

+rsp__model/dense_1/bias/Read/ReadVariableOpReadVariableOprsp__model/dense_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?"
value?"B?" B?"
?
dropout
maxpool
	conv1
	conv2
	conv3
	conv4
flatten
d1
	d2

regularization_losses
trainable_variables
	variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
R
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
h

9kernel
:bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
 
V
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
V
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
?

?layers

regularization_losses
@non_trainable_variables
Alayer_regularization_losses
trainable_variables
Blayer_metrics
Cmetrics
	variables
 
 
 
 
?

Dlayers
regularization_losses
Enon_trainable_variables
Flayer_regularization_losses
trainable_variables
Glayer_metrics
Hmetrics
	variables
 
 
 
?

Ilayers
regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses
trainable_variables
Llayer_metrics
Mmetrics
	variables
US
VARIABLE_VALUErsp__model/conv2d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUErsp__model/conv2d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Nlayers
regularization_losses
Onon_trainable_variables
Player_regularization_losses
trainable_variables
Qlayer_metrics
Rmetrics
	variables
WU
VARIABLE_VALUErsp__model/conv2d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUErsp__model/conv2d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Slayers
regularization_losses
Tnon_trainable_variables
Ulayer_regularization_losses
 trainable_variables
Vlayer_metrics
Wmetrics
!	variables
WU
VARIABLE_VALUErsp__model/conv2d_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUErsp__model/conv2d_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
?

Xlayers
%regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses
&trainable_variables
[layer_metrics
\metrics
'	variables
WU
VARIABLE_VALUErsp__model/conv2d_3/kernel'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUErsp__model/conv2d_3/bias%conv4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?

]layers
+regularization_losses
^non_trainable_variables
_layer_regularization_losses
,trainable_variables
`layer_metrics
ametrics
-	variables
 
 
 
?

blayers
/regularization_losses
cnon_trainable_variables
dlayer_regularization_losses
0trainable_variables
elayer_metrics
fmetrics
1	variables
QO
VARIABLE_VALUErsp__model/dense/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUErsp__model/dense/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
?

glayers
5regularization_losses
hnon_trainable_variables
ilayer_regularization_losses
6trainable_variables
jlayer_metrics
kmetrics
7	variables
SQ
VARIABLE_VALUErsp__model/dense_1/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUErsp__model/dense_1/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
?

llayers
;regularization_losses
mnon_trainable_variables
nlayer_regularization_losses
<trainable_variables
olayer_metrics
pmetrics
=	variables
?
0
1
2
3
4
5
6
7
	8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1rsp__model/conv2d/kernelrsp__model/conv2d/biasrsp__model/conv2d_1/kernelrsp__model/conv2d_1/biasrsp__model/conv2d_2/kernelrsp__model/conv2d_2/biasrsp__model/conv2d_3/kernelrsp__model/conv2d_3/biasrsp__model/dense/kernelrsp__model/dense/biasrsp__model/dense_1/kernelrsp__model/dense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1293802
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,rsp__model/conv2d/kernel/Read/ReadVariableOp*rsp__model/conv2d/bias/Read/ReadVariableOp.rsp__model/conv2d_1/kernel/Read/ReadVariableOp,rsp__model/conv2d_1/bias/Read/ReadVariableOp.rsp__model/conv2d_2/kernel/Read/ReadVariableOp,rsp__model/conv2d_2/bias/Read/ReadVariableOp.rsp__model/conv2d_3/kernel/Read/ReadVariableOp,rsp__model/conv2d_3/bias/Read/ReadVariableOp+rsp__model/dense/kernel/Read/ReadVariableOp)rsp__model/dense/bias/Read/ReadVariableOp-rsp__model/dense_1/kernel/Read/ReadVariableOp+rsp__model/dense_1/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_1294596
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamersp__model/conv2d/kernelrsp__model/conv2d/biasrsp__model/conv2d_1/kernelrsp__model/conv2d_1/biasrsp__model/conv2d_2/kernelrsp__model/conv2d_2/biasrsp__model/conv2d_3/kernelrsp__model/conv2d_3/biasrsp__model/dense/kernelrsp__model/dense/biasrsp__model/dense_1/kernelrsp__model/dense_1/bias*
Tin
2*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_1294642??

?
?
__inference_loss_fn_0_1294493G
Crsp__model_conv2d_kernel_regularizer_square_readvariableop_resource
identity??:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCrsp__model_conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
+rsp__model/conv2d/kernel/Regularizer/SquareSquareBrsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2-
+rsp__model/conv2d/kernel/Regularizer/Square?
*rsp__model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*rsp__model/conv2d/kernel/Regularizer/Const?
(rsp__model/conv2d/kernel/Regularizer/SumSum/rsp__model/conv2d/kernel/Regularizer/Square:y:03rsp__model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/Sum?
*rsp__model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*rsp__model/conv2d/kernel/Regularizer/mul/x?
(rsp__model/conv2d/kernel/Regularizer/mulMul3rsp__model/conv2d/kernel/Regularizer/mul/x:output:01rsp__model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/mul?
IdentityIdentity,rsp__model/conv2d/kernel/Regularizer/mul:z:0;^rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_1294425

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_conv2d_1_layer_call_fn_1294355

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_12933222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
E
)__inference_flatten_layer_call_fn_1294430

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_12934132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_1294526I
Ersp__model_conv2d_3_kernel_regularizer_square_readvariableop_resource
identity??<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpErsp__model_conv2d_3_kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype02>
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_3/kernel/Regularizer/SquareSquareDrsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2/
-rsp__model/conv2d_3/kernel/Regularizer/Square?
,rsp__model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_3/kernel/Regularizer/Const?
*rsp__model/conv2d_3/kernel/Regularizer/SumSum1rsp__model/conv2d_3/kernel/Regularizer/Square:y:05rsp__model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/Sum?
,rsp__model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_3/kernel/Regularizer/mul/x?
*rsp__model/conv2d_3/kernel/Regularizer/mulMul5rsp__model/conv2d_3/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/mul?
IdentityIdentity.rsp__model/conv2d_3/kernel/Regularizer/mul:z:0=^rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp
?S
?	
"__inference__wrapped_model_1293254
input_14
0rsp__model_conv2d_conv2d_readvariableop_resource5
1rsp__model_conv2d_biasadd_readvariableop_resource6
2rsp__model_conv2d_1_conv2d_readvariableop_resource7
3rsp__model_conv2d_1_biasadd_readvariableop_resource6
2rsp__model_conv2d_2_conv2d_readvariableop_resource7
3rsp__model_conv2d_2_biasadd_readvariableop_resource6
2rsp__model_conv2d_3_conv2d_readvariableop_resource7
3rsp__model_conv2d_3_biasadd_readvariableop_resource3
/rsp__model_dense_matmul_readvariableop_resource4
0rsp__model_dense_biasadd_readvariableop_resource5
1rsp__model_dense_1_matmul_readvariableop_resource6
2rsp__model_dense_1_biasadd_readvariableop_resource
identity??(rsp__model/conv2d/BiasAdd/ReadVariableOp?'rsp__model/conv2d/Conv2D/ReadVariableOp?*rsp__model/conv2d_1/BiasAdd/ReadVariableOp?)rsp__model/conv2d_1/Conv2D/ReadVariableOp?*rsp__model/conv2d_2/BiasAdd/ReadVariableOp?)rsp__model/conv2d_2/Conv2D/ReadVariableOp?*rsp__model/conv2d_3/BiasAdd/ReadVariableOp?)rsp__model/conv2d_3/Conv2D/ReadVariableOp?'rsp__model/dense/BiasAdd/ReadVariableOp?&rsp__model/dense/MatMul/ReadVariableOp?)rsp__model/dense_1/BiasAdd/ReadVariableOp?(rsp__model/dense_1/MatMul/ReadVariableOp?
'rsp__model/conv2d/Conv2D/ReadVariableOpReadVariableOp0rsp__model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'rsp__model/conv2d/Conv2D/ReadVariableOp?
rsp__model/conv2d/Conv2DConv2Dinput_1/rsp__model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@*
paddingVALID*
strides
2
rsp__model/conv2d/Conv2D?
(rsp__model/conv2d/BiasAdd/ReadVariableOpReadVariableOp1rsp__model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(rsp__model/conv2d/BiasAdd/ReadVariableOp?
rsp__model/conv2d/BiasAddBiasAdd!rsp__model/conv2d/Conv2D:output:00rsp__model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@2
rsp__model/conv2d/BiasAdd?
rsp__model/conv2d/ReluRelu"rsp__model/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????>>@2
rsp__model/conv2d/Relu?
 rsp__model/max_pooling2d/MaxPoolMaxPool$rsp__model/conv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2"
 rsp__model/max_pooling2d/MaxPool?
)rsp__model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2rsp__model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)rsp__model/conv2d_1/Conv2D/ReadVariableOp?
rsp__model/conv2d_1/Conv2DConv2D)rsp__model/max_pooling2d/MaxPool:output:01rsp__model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
rsp__model/conv2d_1/Conv2D?
*rsp__model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3rsp__model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*rsp__model/conv2d_1/BiasAdd/ReadVariableOp?
rsp__model/conv2d_1/BiasAddBiasAdd#rsp__model/conv2d_1/Conv2D:output:02rsp__model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
rsp__model/conv2d_1/BiasAdd?
rsp__model/conv2d_1/ReluRelu$rsp__model/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
rsp__model/conv2d_1/Relu?
"rsp__model/max_pooling2d/MaxPool_1MaxPool&rsp__model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2$
"rsp__model/max_pooling2d/MaxPool_1?
)rsp__model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2rsp__model_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02+
)rsp__model/conv2d_2/Conv2D/ReadVariableOp?
rsp__model/conv2d_2/Conv2DConv2D+rsp__model/max_pooling2d/MaxPool_1:output:01rsp__model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
rsp__model/conv2d_2/Conv2D?
*rsp__model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3rsp__model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*rsp__model/conv2d_2/BiasAdd/ReadVariableOp?
rsp__model/conv2d_2/BiasAddBiasAdd#rsp__model/conv2d_2/Conv2D:output:02rsp__model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
rsp__model/conv2d_2/BiasAdd?
rsp__model/conv2d_2/ReluRelu$rsp__model/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
rsp__model/conv2d_2/Relu?
"rsp__model/max_pooling2d/MaxPool_2MaxPool&rsp__model/conv2d_2/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2$
"rsp__model/max_pooling2d/MaxPool_2?
)rsp__model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2rsp__model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02+
)rsp__model/conv2d_3/Conv2D/ReadVariableOp?
rsp__model/conv2d_3/Conv2DConv2D+rsp__model/max_pooling2d/MaxPool_2:output:01rsp__model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
rsp__model/conv2d_3/Conv2D?
*rsp__model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3rsp__model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*rsp__model/conv2d_3/BiasAdd/ReadVariableOp?
rsp__model/conv2d_3/BiasAddBiasAdd#rsp__model/conv2d_3/Conv2D:output:02rsp__model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
rsp__model/conv2d_3/BiasAdd?
rsp__model/conv2d_3/ReluRelu$rsp__model/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
rsp__model/conv2d_3/Relu?
"rsp__model/max_pooling2d/MaxPool_3MaxPool&rsp__model/conv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2$
"rsp__model/max_pooling2d/MaxPool_3?
rsp__model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
rsp__model/flatten/Const?
rsp__model/flatten/ReshapeReshape+rsp__model/max_pooling2d/MaxPool_3:output:0!rsp__model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
rsp__model/flatten/Reshape?
rsp__model/dropout/IdentityIdentity#rsp__model/flatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
rsp__model/dropout/Identity?
&rsp__model/dense/MatMul/ReadVariableOpReadVariableOp/rsp__model_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&rsp__model/dense/MatMul/ReadVariableOp?
rsp__model/dense/MatMulMatMul$rsp__model/dropout/Identity:output:0.rsp__model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
rsp__model/dense/MatMul?
'rsp__model/dense/BiasAdd/ReadVariableOpReadVariableOp0rsp__model_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'rsp__model/dense/BiasAdd/ReadVariableOp?
rsp__model/dense/BiasAddBiasAdd!rsp__model/dense/MatMul:product:0/rsp__model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
rsp__model/dense/BiasAdd?
rsp__model/dense/ReluRelu!rsp__model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
rsp__model/dense/Relu?
(rsp__model/dense_1/MatMul/ReadVariableOpReadVariableOp1rsp__model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(rsp__model/dense_1/MatMul/ReadVariableOp?
rsp__model/dense_1/MatMulMatMul#rsp__model/dense/Relu:activations:00rsp__model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rsp__model/dense_1/MatMul?
)rsp__model/dense_1/BiasAdd/ReadVariableOpReadVariableOp2rsp__model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rsp__model/dense_1/BiasAdd/ReadVariableOp?
rsp__model/dense_1/BiasAddBiasAdd#rsp__model/dense_1/MatMul:product:01rsp__model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rsp__model/dense_1/BiasAdd?
rsp__model/dense_1/SoftmaxSoftmax#rsp__model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
rsp__model/dense_1/Softmax?
IdentityIdentity$rsp__model/dense_1/Softmax:softmax:0)^rsp__model/conv2d/BiasAdd/ReadVariableOp(^rsp__model/conv2d/Conv2D/ReadVariableOp+^rsp__model/conv2d_1/BiasAdd/ReadVariableOp*^rsp__model/conv2d_1/Conv2D/ReadVariableOp+^rsp__model/conv2d_2/BiasAdd/ReadVariableOp*^rsp__model/conv2d_2/Conv2D/ReadVariableOp+^rsp__model/conv2d_3/BiasAdd/ReadVariableOp*^rsp__model/conv2d_3/Conv2D/ReadVariableOp(^rsp__model/dense/BiasAdd/ReadVariableOp'^rsp__model/dense/MatMul/ReadVariableOp*^rsp__model/dense_1/BiasAdd/ReadVariableOp)^rsp__model/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::2T
(rsp__model/conv2d/BiasAdd/ReadVariableOp(rsp__model/conv2d/BiasAdd/ReadVariableOp2R
'rsp__model/conv2d/Conv2D/ReadVariableOp'rsp__model/conv2d/Conv2D/ReadVariableOp2X
*rsp__model/conv2d_1/BiasAdd/ReadVariableOp*rsp__model/conv2d_1/BiasAdd/ReadVariableOp2V
)rsp__model/conv2d_1/Conv2D/ReadVariableOp)rsp__model/conv2d_1/Conv2D/ReadVariableOp2X
*rsp__model/conv2d_2/BiasAdd/ReadVariableOp*rsp__model/conv2d_2/BiasAdd/ReadVariableOp2V
)rsp__model/conv2d_2/Conv2D/ReadVariableOp)rsp__model/conv2d_2/Conv2D/ReadVariableOp2X
*rsp__model/conv2d_3/BiasAdd/ReadVariableOp*rsp__model/conv2d_3/BiasAdd/ReadVariableOp2V
)rsp__model/conv2d_3/Conv2D/ReadVariableOp)rsp__model/conv2d_3/Conv2D/ReadVariableOp2R
'rsp__model/dense/BiasAdd/ReadVariableOp'rsp__model/dense/BiasAdd/ReadVariableOp2P
&rsp__model/dense/MatMul/ReadVariableOp&rsp__model/dense/MatMul/ReadVariableOp2V
)rsp__model/dense_1/BiasAdd/ReadVariableOp)rsp__model/dense_1/BiasAdd/ReadVariableOp2T
(rsp__model/dense_1/MatMul/ReadVariableOp(rsp__model/dense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_1293413

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
,__inference_rsp__model_layer_call_fn_1294264
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_rsp__model_layer_call_and_return_conditional_losses_12936852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????@@

_user_specified_namex
?	
?
,__inference_rsp__model_layer_call_fn_1294004
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_rsp__model_layer_call_and_return_conditional_losses_12936852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1293322

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02>
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_1/kernel/Regularizer/SquareSquareDrsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2/
-rsp__model/conv2d_1/kernel/Regularizer/Square?
,rsp__model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_1/kernel/Regularizer/Const?
*rsp__model/conv2d_1/kernel/Regularizer/SumSum1rsp__model/conv2d_1/kernel/Regularizer/Square:y:05rsp__model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/Sum?
,rsp__model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_1/kernel/Regularizer/mul/x?
*rsp__model/conv2d_1/kernel/Regularizer/mulMul5rsp__model/conv2d_1/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_layer_call_fn_1293267

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_12932612
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_layer_call_and_return_conditional_losses_1294281

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_layer_call_and_return_conditional_losses_1293288

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????>>@2
Relu?
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
+rsp__model/conv2d/kernel/Regularizer/SquareSquareBrsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2-
+rsp__model/conv2d/kernel/Regularizer/Square?
*rsp__model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*rsp__model/conv2d/kernel/Regularizer/Const?
(rsp__model/conv2d/kernel/Regularizer/SumSum/rsp__model/conv2d/kernel/Regularizer/Square:y:03rsp__model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/Sum?
*rsp__model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*rsp__model/conv2d/kernel/Regularizer/mul/x?
(rsp__model/conv2d/kernel/Regularizer/mulMul3rsp__model/conv2d/kernel/Regularizer/mul/x:output:01rsp__model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????>>@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
,__inference_rsp__model_layer_call_fn_1294033
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_rsp__model_layer_call_and_return_conditional_losses_12936852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1294346

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02>
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_1/kernel/Regularizer/SquareSquareDrsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2/
-rsp__model/conv2d_1/kernel/Regularizer/Square?
,rsp__model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_1/kernel/Regularizer/Const?
*rsp__model/conv2d_1/kernel/Regularizer/SumSum1rsp__model/conv2d_1/kernel/Regularizer/Square:y:05rsp__model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/Sum?
,rsp__model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_1/kernel/Regularizer/mul/x?
*rsp__model/conv2d_1/kernel/Regularizer/mulMul5rsp__model/conv2d_1/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_dense_layer_call_and_return_conditional_losses_1293468

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
*rsp__model/dense/kernel/Regularizer/SquareSquareArsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2,
*rsp__model/dense/kernel/Regularizer/Square?
)rsp__model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)rsp__model/dense/kernel/Regularizer/Const?
'rsp__model/dense/kernel/Regularizer/SumSum.rsp__model/dense/kernel/Regularizer/Square:y:02rsp__model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/Sum?
)rsp__model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)rsp__model/dense/kernel/Regularizer/mul/x?
'rsp__model/dense/kernel/Regularizer/mulMul2rsp__model/dense/kernel/Regularizer/mul/x:output:00rsp__model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp:^rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2v
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_1294515I
Ersp__model_conv2d_2_kernel_regularizer_square_readvariableop_resource
identity??<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpErsp__model_conv2d_2_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@?*
dtype02>
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_2/kernel/Regularizer/SquareSquareDrsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?2/
-rsp__model/conv2d_2/kernel/Regularizer/Square?
,rsp__model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_2/kernel/Regularizer/Const?
*rsp__model/conv2d_2/kernel/Regularizer/SumSum1rsp__model/conv2d_2/kernel/Regularizer/Square:y:05rsp__model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/Sum?
,rsp__model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_2/kernel/Regularizer/mul/x?
*rsp__model/conv2d_2/kernel/Regularizer/mulMul5rsp__model/conv2d_2/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/mul?
IdentityIdentity.rsp__model/conv2d_2/kernel/Regularizer/mul:z:0=^rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp
?
?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1293390

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02>
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_3/kernel/Regularizer/SquareSquareDrsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2/
-rsp__model/conv2d_3/kernel/Regularizer/Square?
,rsp__model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_3/kernel/Regularizer/Const?
*rsp__model/conv2d_3/kernel/Regularizer/SumSum1rsp__model/conv2d_3/kernel/Regularizer/Square:y:05rsp__model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/Sum?
,rsp__model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_3/kernel/Regularizer/mul/x?
*rsp__model/conv2d_3/kernel/Regularizer/mulMul5rsp__model/conv2d_3/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_layer_call_and_return_conditional_losses_1294314

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????>>@2
Relu?
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
+rsp__model/conv2d/kernel/Regularizer/SquareSquareBrsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2-
+rsp__model/conv2d/kernel/Regularizer/Square?
*rsp__model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*rsp__model/conv2d/kernel/Regularizer/Const?
(rsp__model/conv2d/kernel/Regularizer/SumSum/rsp__model/conv2d/kernel/Regularizer/Square:y:03rsp__model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/Sum?
*rsp__model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*rsp__model/conv2d/kernel/Regularizer/mul/x?
(rsp__model/conv2d/kernel/Regularizer/mulMul3rsp__model/conv2d/kernel/Regularizer/mul/x:output:01rsp__model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????>>@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?

G__inference_rsp__model_layer_call_and_return_conditional_losses_1293892
input_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????>>@2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_2/Relu?
max_pooling2d/MaxPool_2MaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_2?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D max_pooling2d/MaxPool_2:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_3/Relu?
max_pooling2d/MaxPool_3MaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_3o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape max_pooling2d/MaxPool_3:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
+rsp__model/conv2d/kernel/Regularizer/SquareSquareBrsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2-
+rsp__model/conv2d/kernel/Regularizer/Square?
*rsp__model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*rsp__model/conv2d/kernel/Regularizer/Const?
(rsp__model/conv2d/kernel/Regularizer/SumSum/rsp__model/conv2d/kernel/Regularizer/Square:y:03rsp__model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/Sum?
*rsp__model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*rsp__model/conv2d/kernel/Regularizer/mul/x?
(rsp__model/conv2d/kernel/Regularizer/mulMul3rsp__model/conv2d/kernel/Regularizer/mul/x:output:01rsp__model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/mul?
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02>
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_1/kernel/Regularizer/SquareSquareDrsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2/
-rsp__model/conv2d_1/kernel/Regularizer/Square?
,rsp__model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_1/kernel/Regularizer/Const?
*rsp__model/conv2d_1/kernel/Regularizer/SumSum1rsp__model/conv2d_1/kernel/Regularizer/Square:y:05rsp__model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/Sum?
,rsp__model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_1/kernel/Regularizer/mul/x?
*rsp__model/conv2d_1/kernel/Regularizer/mulMul5rsp__model/conv2d_1/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/mul?
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02>
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_2/kernel/Regularizer/SquareSquareDrsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?2/
-rsp__model/conv2d_2/kernel/Regularizer/Square?
,rsp__model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_2/kernel/Regularizer/Const?
*rsp__model/conv2d_2/kernel/Regularizer/SumSum1rsp__model/conv2d_2/kernel/Regularizer/Square:y:05rsp__model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/Sum?
,rsp__model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_2/kernel/Regularizer/mul/x?
*rsp__model/conv2d_2/kernel/Regularizer/mulMul5rsp__model/conv2d_2/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/mul?
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02>
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_3/kernel/Regularizer/SquareSquareDrsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2/
-rsp__model/conv2d_3/kernel/Regularizer/Square?
,rsp__model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_3/kernel/Regularizer/Const?
*rsp__model/conv2d_3/kernel/Regularizer/SumSum1rsp__model/conv2d_3/kernel/Regularizer/Square:y:05rsp__model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/Sum?
,rsp__model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_3/kernel/Regularizer/mul/x?
*rsp__model/conv2d_3/kernel/Regularizer/mulMul5rsp__model/conv2d_3/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/mul?
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
*rsp__model/dense/kernel/Regularizer/SquareSquareArsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2,
*rsp__model/dense/kernel/Regularizer/Square?
)rsp__model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)rsp__model/dense/kernel/Regularizer/Const?
'rsp__model/dense/kernel/Regularizer/SumSum.rsp__model/dense/kernel/Regularizer/Square:y:02rsp__model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/Sum?
)rsp__model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)rsp__model/dense/kernel/Regularizer/mul/x?
'rsp__model/dense/kernel/Regularizer/mulMul2rsp__model/dense/kernel/Regularizer/mul/x:output:00rsp__model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/mul?
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp;^rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:^rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2x
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2v
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_1293495

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_layer_call_fn_1294323

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>>@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_12932882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????>>@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
%__inference_signature_wrapper_1293802
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_12932542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?	
?
,__inference_rsp__model_layer_call_fn_1294235
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_rsp__model_layer_call_and_return_conditional_losses_12936852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????@@

_user_specified_namex
?
?

G__inference_rsp__model_layer_call_and_return_conditional_losses_1293975
input_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????>>@2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_2/Relu?
max_pooling2d/MaxPool_2MaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_2?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D max_pooling2d/MaxPool_2:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_3/Relu?
max_pooling2d/MaxPool_3MaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_3o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape max_pooling2d/MaxPool_3:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
+rsp__model/conv2d/kernel/Regularizer/SquareSquareBrsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2-
+rsp__model/conv2d/kernel/Regularizer/Square?
*rsp__model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*rsp__model/conv2d/kernel/Regularizer/Const?
(rsp__model/conv2d/kernel/Regularizer/SumSum/rsp__model/conv2d/kernel/Regularizer/Square:y:03rsp__model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/Sum?
*rsp__model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*rsp__model/conv2d/kernel/Regularizer/mul/x?
(rsp__model/conv2d/kernel/Regularizer/mulMul3rsp__model/conv2d/kernel/Regularizer/mul/x:output:01rsp__model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/mul?
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02>
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_1/kernel/Regularizer/SquareSquareDrsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2/
-rsp__model/conv2d_1/kernel/Regularizer/Square?
,rsp__model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_1/kernel/Regularizer/Const?
*rsp__model/conv2d_1/kernel/Regularizer/SumSum1rsp__model/conv2d_1/kernel/Regularizer/Square:y:05rsp__model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/Sum?
,rsp__model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_1/kernel/Regularizer/mul/x?
*rsp__model/conv2d_1/kernel/Regularizer/mulMul5rsp__model/conv2d_1/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/mul?
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02>
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_2/kernel/Regularizer/SquareSquareDrsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?2/
-rsp__model/conv2d_2/kernel/Regularizer/Square?
,rsp__model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_2/kernel/Regularizer/Const?
*rsp__model/conv2d_2/kernel/Regularizer/SumSum1rsp__model/conv2d_2/kernel/Regularizer/Square:y:05rsp__model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/Sum?
,rsp__model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_2/kernel/Regularizer/mul/x?
*rsp__model/conv2d_2/kernel/Regularizer/mulMul5rsp__model/conv2d_2/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/mul?
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02>
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_3/kernel/Regularizer/SquareSquareDrsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2/
-rsp__model/conv2d_3/kernel/Regularizer/Square?
,rsp__model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_3/kernel/Regularizer/Const?
*rsp__model/conv2d_3/kernel/Regularizer/SumSum1rsp__model/conv2d_3/kernel/Regularizer/Square:y:05rsp__model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/Sum?
,rsp__model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_3/kernel/Regularizer/mul/x?
*rsp__model/conv2d_3/kernel/Regularizer/mulMul5rsp__model/conv2d_3/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/mul?
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
*rsp__model/dense/kernel/Regularizer/SquareSquareArsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2,
*rsp__model/dense/kernel/Regularizer/Square?
)rsp__model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)rsp__model/dense/kernel/Regularizer/Const?
'rsp__model/dense/kernel/Regularizer/SumSum.rsp__model/dense/kernel/Regularizer/Square:y:02rsp__model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/Sum?
)rsp__model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)rsp__model/dense/kernel/Regularizer/mul/x?
'rsp__model/dense/kernel/Regularizer/mulMul2rsp__model/dense/kernel/Regularizer/mul/x:output:00rsp__model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/mul?
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp;^rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:^rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2x
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2v
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?~
?

G__inference_rsp__model_layer_call_and_return_conditional_losses_1294206
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????>>@2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_2/Relu?
max_pooling2d/MaxPool_2MaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_2?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D max_pooling2d/MaxPool_2:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_3/Relu?
max_pooling2d/MaxPool_3MaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_3o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape max_pooling2d/MaxPool_3:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
+rsp__model/conv2d/kernel/Regularizer/SquareSquareBrsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2-
+rsp__model/conv2d/kernel/Regularizer/Square?
*rsp__model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*rsp__model/conv2d/kernel/Regularizer/Const?
(rsp__model/conv2d/kernel/Regularizer/SumSum/rsp__model/conv2d/kernel/Regularizer/Square:y:03rsp__model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/Sum?
*rsp__model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*rsp__model/conv2d/kernel/Regularizer/mul/x?
(rsp__model/conv2d/kernel/Regularizer/mulMul3rsp__model/conv2d/kernel/Regularizer/mul/x:output:01rsp__model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/mul?
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02>
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_1/kernel/Regularizer/SquareSquareDrsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2/
-rsp__model/conv2d_1/kernel/Regularizer/Square?
,rsp__model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_1/kernel/Regularizer/Const?
*rsp__model/conv2d_1/kernel/Regularizer/SumSum1rsp__model/conv2d_1/kernel/Regularizer/Square:y:05rsp__model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/Sum?
,rsp__model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_1/kernel/Regularizer/mul/x?
*rsp__model/conv2d_1/kernel/Regularizer/mulMul5rsp__model/conv2d_1/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/mul?
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02>
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_2/kernel/Regularizer/SquareSquareDrsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?2/
-rsp__model/conv2d_2/kernel/Regularizer/Square?
,rsp__model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_2/kernel/Regularizer/Const?
*rsp__model/conv2d_2/kernel/Regularizer/SumSum1rsp__model/conv2d_2/kernel/Regularizer/Square:y:05rsp__model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/Sum?
,rsp__model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_2/kernel/Regularizer/mul/x?
*rsp__model/conv2d_2/kernel/Regularizer/mulMul5rsp__model/conv2d_2/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/mul?
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02>
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_3/kernel/Regularizer/SquareSquareDrsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2/
-rsp__model/conv2d_3/kernel/Regularizer/Square?
,rsp__model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_3/kernel/Regularizer/Const?
*rsp__model/conv2d_3/kernel/Regularizer/SumSum1rsp__model/conv2d_3/kernel/Regularizer/Square:y:05rsp__model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/Sum?
,rsp__model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_3/kernel/Regularizer/mul/x?
*rsp__model/conv2d_3/kernel/Regularizer/mulMul5rsp__model/conv2d_3/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/mul?
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
*rsp__model/dense/kernel/Regularizer/SquareSquareArsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2,
*rsp__model/dense/kernel/Regularizer/Square?
)rsp__model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)rsp__model/dense/kernel/Regularizer/Const?
'rsp__model/dense/kernel/Regularizer/SumSum.rsp__model/dense/kernel/Regularizer/Square:y:02rsp__model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/Sum?
)rsp__model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)rsp__model/dense/kernel/Regularizer/mul/x?
'rsp__model/dense/kernel/Regularizer/mulMul2rsp__model/dense/kernel/Regularizer/mul/x:output:00rsp__model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/mul?
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp;^rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:^rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2x
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2v
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????@@

_user_specified_namex
?
c
D__inference_dropout_layer_call_and_return_conditional_losses_1294276

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_conv2d_2_layer_call_fn_1294387

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_12933562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1293261

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1293356

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02>
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_2/kernel/Regularizer/SquareSquareDrsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?2/
-rsp__model/conv2d_2/kernel/Regularizer/Square?
,rsp__model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_2/kernel/Regularizer/Const?
*rsp__model/conv2d_2/kernel/Regularizer/SumSum1rsp__model/conv2d_2/kernel/Regularizer/Square:y:05rsp__model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/Sum?
,rsp__model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_2/kernel/Regularizer/mul/x?
*rsp__model/conv2d_2/kernel/Regularizer/mulMul5rsp__model/conv2d_2/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

*__inference_conv2d_3_layer_call_fn_1294419

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_12933902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_layer_call_fn_1294291

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_12934382
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_1294537F
Brsp__model_dense_kernel_regularizer_square_readvariableop_resource
identity??9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBrsp__model_dense_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
*rsp__model/dense/kernel/Regularizer/SquareSquareArsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2,
*rsp__model/dense/kernel/Regularizer/Square?
)rsp__model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)rsp__model/dense/kernel/Regularizer/Const?
'rsp__model/dense/kernel/Regularizer/SumSum.rsp__model/dense/kernel/Regularizer/Square:y:02rsp__model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/Sum?
)rsp__model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)rsp__model/dense/kernel/Regularizer/mul/x?
'rsp__model/dense/kernel/Regularizer/mulMul2rsp__model/dense/kernel/Regularizer/mul/x:output:00rsp__model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/mul?
IdentityIdentity+rsp__model/dense/kernel/Regularizer/mul:z:0:^rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2v
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp
?
b
)__inference_dropout_layer_call_fn_1294286

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_12934332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_layer_call_and_return_conditional_losses_1293433

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1294410

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02>
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_3/kernel/Regularizer/SquareSquareDrsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2/
-rsp__model/conv2d_3/kernel/Regularizer/Square?
,rsp__model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_3/kernel/Regularizer/Const?
*rsp__model/conv2d_3/kernel/Regularizer/SumSum1rsp__model/conv2d_3/kernel/Regularizer/Square:y:05rsp__model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/Sum?
,rsp__model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_3/kernel/Regularizer/mul/x?
*rsp__model/conv2d_3/kernel/Regularizer/mulMul5rsp__model/conv2d_3/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_dense_layer_call_and_return_conditional_losses_1294453

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
*rsp__model/dense/kernel/Regularizer/SquareSquareArsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2,
*rsp__model/dense/kernel/Regularizer/Square?
)rsp__model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)rsp__model/dense/kernel/Regularizer/Const?
'rsp__model/dense/kernel/Regularizer/SumSum.rsp__model/dense/kernel/Regularizer/Square:y:02rsp__model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/Sum?
)rsp__model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)rsp__model/dense/kernel/Regularizer/mul/x?
'rsp__model/dense/kernel/Regularizer/mulMul2rsp__model/dense/kernel/Regularizer/mul/x:output:00rsp__model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp:^rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2v
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_1294473

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_1294504I
Ersp__model_conv2d_1_kernel_regularizer_square_readvariableop_resource
identity??<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpErsp__model_conv2d_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype02>
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_1/kernel/Regularizer/SquareSquareDrsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2/
-rsp__model/conv2d_1/kernel/Regularizer/Square?
,rsp__model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_1/kernel/Regularizer/Const?
*rsp__model/conv2d_1/kernel/Regularizer/SumSum1rsp__model/conv2d_1/kernel/Regularizer/Square:y:05rsp__model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/Sum?
,rsp__model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_1/kernel/Regularizer/mul/x?
*rsp__model/conv2d_1/kernel/Regularizer/mulMul5rsp__model/conv2d_1/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/mul?
IdentityIdentity.rsp__model/conv2d_1/kernel/Regularizer/mul:z:0=^rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp
?
|
'__inference_dense_layer_call_fn_1294462

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_12934682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?

G__inference_rsp__model_layer_call_and_return_conditional_losses_1294123
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>>@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????>>@2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_2/Relu?
max_pooling2d/MaxPool_2MaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_2?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D max_pooling2d/MaxPool_2:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_3/Relu?
max_pooling2d/MaxPool_3MaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_3o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape max_pooling2d/MaxPool_3:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
+rsp__model/conv2d/kernel/Regularizer/SquareSquareBrsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2-
+rsp__model/conv2d/kernel/Regularizer/Square?
*rsp__model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*rsp__model/conv2d/kernel/Regularizer/Const?
(rsp__model/conv2d/kernel/Regularizer/SumSum/rsp__model/conv2d/kernel/Regularizer/Square:y:03rsp__model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/Sum?
*rsp__model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*rsp__model/conv2d/kernel/Regularizer/mul/x?
(rsp__model/conv2d/kernel/Regularizer/mulMul3rsp__model/conv2d/kernel/Regularizer/mul/x:output:01rsp__model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/mul?
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02>
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_1/kernel/Regularizer/SquareSquareDrsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2/
-rsp__model/conv2d_1/kernel/Regularizer/Square?
,rsp__model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_1/kernel/Regularizer/Const?
*rsp__model/conv2d_1/kernel/Regularizer/SumSum1rsp__model/conv2d_1/kernel/Regularizer/Square:y:05rsp__model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/Sum?
,rsp__model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_1/kernel/Regularizer/mul/x?
*rsp__model/conv2d_1/kernel/Regularizer/mulMul5rsp__model/conv2d_1/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/mul?
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02>
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_2/kernel/Regularizer/SquareSquareDrsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?2/
-rsp__model/conv2d_2/kernel/Regularizer/Square?
,rsp__model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_2/kernel/Regularizer/Const?
*rsp__model/conv2d_2/kernel/Regularizer/SumSum1rsp__model/conv2d_2/kernel/Regularizer/Square:y:05rsp__model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/Sum?
,rsp__model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_2/kernel/Regularizer/mul/x?
*rsp__model/conv2d_2/kernel/Regularizer/mulMul5rsp__model/conv2d_2/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/mul?
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02>
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_3/kernel/Regularizer/SquareSquareDrsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2/
-rsp__model/conv2d_3/kernel/Regularizer/Square?
,rsp__model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_3/kernel/Regularizer/Const?
*rsp__model/conv2d_3/kernel/Regularizer/SumSum1rsp__model/conv2d_3/kernel/Regularizer/Square:y:05rsp__model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/Sum?
,rsp__model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_3/kernel/Regularizer/mul/x?
*rsp__model/conv2d_3/kernel/Regularizer/mulMul5rsp__model/conv2d_3/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/mul?
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
*rsp__model/dense/kernel/Regularizer/SquareSquareArsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2,
*rsp__model/dense/kernel/Regularizer/Square?
)rsp__model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)rsp__model/dense/kernel/Regularizer/Const?
'rsp__model/dense/kernel/Regularizer/SumSum.rsp__model/dense/kernel/Regularizer/Square:y:02rsp__model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/Sum?
)rsp__model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)rsp__model/dense/kernel/Regularizer/mul/x?
'rsp__model/dense/kernel/Regularizer/mulMul2rsp__model/dense/kernel/Regularizer/mul/x:output:00rsp__model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/mul?
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp;^rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:^rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2x
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2v
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????@@

_user_specified_namex
?%
?
 __inference__traced_save_1294596
file_prefix7
3savev2_rsp__model_conv2d_kernel_read_readvariableop5
1savev2_rsp__model_conv2d_bias_read_readvariableop9
5savev2_rsp__model_conv2d_1_kernel_read_readvariableop7
3savev2_rsp__model_conv2d_1_bias_read_readvariableop9
5savev2_rsp__model_conv2d_2_kernel_read_readvariableop7
3savev2_rsp__model_conv2d_2_bias_read_readvariableop9
5savev2_rsp__model_conv2d_3_kernel_read_readvariableop7
3savev2_rsp__model_conv2d_3_bias_read_readvariableop6
2savev2_rsp__model_dense_kernel_read_readvariableop4
0savev2_rsp__model_dense_bias_read_readvariableop8
4savev2_rsp__model_dense_1_kernel_read_readvariableop6
2savev2_rsp__model_dense_1_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_rsp__model_conv2d_kernel_read_readvariableop1savev2_rsp__model_conv2d_bias_read_readvariableop5savev2_rsp__model_conv2d_1_kernel_read_readvariableop3savev2_rsp__model_conv2d_1_bias_read_readvariableop5savev2_rsp__model_conv2d_2_kernel_read_readvariableop3savev2_rsp__model_conv2d_2_bias_read_readvariableop5savev2_rsp__model_conv2d_3_kernel_read_readvariableop3savev2_rsp__model_conv2d_3_bias_read_readvariableop2savev2_rsp__model_dense_kernel_read_readvariableop0savev2_rsp__model_dense_bias_read_readvariableop4savev2_rsp__model_dense_1_kernel_read_readvariableop2savev2_rsp__model_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@?:?:??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?
b
D__inference_dropout_layer_call_and_return_conditional_losses_1293438

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?m
?
G__inference_rsp__model_layer_call_and_return_conditional_losses_1293685
x
conv2d_1293618
conv2d_1293620
conv2d_1_1293624
conv2d_1_1293626
conv2d_2_1293630
conv2d_2_1293632
conv2d_3_1293636
conv2d_3_1293638
dense_1293644
dense_1293646
dense_1_1293649
dense_1_1293651
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_1293618conv2d_1293620*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>>@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_12932882 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_12932612
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1293624conv2d_1_1293626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_12933222"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d/PartitionedCall_1PartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_12932612!
max_pooling2d/PartitionedCall_1?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d/PartitionedCall_1:output:0conv2d_2_1293630conv2d_2_1293632*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_12933562"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d/PartitionedCall_2PartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_12932612!
max_pooling2d/PartitionedCall_2?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d/PartitionedCall_2:output:0conv2d_3_1293636conv2d_3_1293638*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_12933902"
 conv2d_3/StatefulPartitionedCall?
max_pooling2d/PartitionedCall_3PartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_12932612!
max_pooling2d/PartitionedCall_3?
flatten/PartitionedCallPartitionedCall(max_pooling2d/PartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_12934132
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_12934382
dropout/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1293644dense_1293646*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_12934682
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1293649dense_1_1293651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_12934952!
dense_1/StatefulPartitionedCall?
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1293618*&
_output_shapes
:@*
dtype02<
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
+rsp__model/conv2d/kernel/Regularizer/SquareSquareBrsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2-
+rsp__model/conv2d/kernel/Regularizer/Square?
*rsp__model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*rsp__model/conv2d/kernel/Regularizer/Const?
(rsp__model/conv2d/kernel/Regularizer/SumSum/rsp__model/conv2d/kernel/Regularizer/Square:y:03rsp__model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/Sum?
*rsp__model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*rsp__model/conv2d/kernel/Regularizer/mul/x?
(rsp__model/conv2d/kernel/Regularizer/mulMul3rsp__model/conv2d/kernel/Regularizer/mul/x:output:01rsp__model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(rsp__model/conv2d/kernel/Regularizer/mul?
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_1293624*&
_output_shapes
:@@*
dtype02>
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_1/kernel/Regularizer/SquareSquareDrsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2/
-rsp__model/conv2d_1/kernel/Regularizer/Square?
,rsp__model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_1/kernel/Regularizer/Const?
*rsp__model/conv2d_1/kernel/Regularizer/SumSum1rsp__model/conv2d_1/kernel/Regularizer/Square:y:05rsp__model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/Sum?
,rsp__model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_1/kernel/Regularizer/mul/x?
*rsp__model/conv2d_1/kernel/Regularizer/mulMul5rsp__model/conv2d_1/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_1/kernel/Regularizer/mul?
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_1293630*'
_output_shapes
:@?*
dtype02>
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_2/kernel/Regularizer/SquareSquareDrsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?2/
-rsp__model/conv2d_2/kernel/Regularizer/Square?
,rsp__model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_2/kernel/Regularizer/Const?
*rsp__model/conv2d_2/kernel/Regularizer/SumSum1rsp__model/conv2d_2/kernel/Regularizer/Square:y:05rsp__model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/Sum?
,rsp__model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_2/kernel/Regularizer/mul/x?
*rsp__model/conv2d_2/kernel/Regularizer/mulMul5rsp__model/conv2d_2/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/mul?
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_1293636*(
_output_shapes
:??*
dtype02>
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_3/kernel/Regularizer/SquareSquareDrsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2/
-rsp__model/conv2d_3/kernel/Regularizer/Square?
,rsp__model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_3/kernel/Regularizer/Const?
*rsp__model/conv2d_3/kernel/Regularizer/SumSum1rsp__model/conv2d_3/kernel/Regularizer/Square:y:05rsp__model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/Sum?
,rsp__model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_3/kernel/Regularizer/mul/x?
*rsp__model/conv2d_3/kernel/Regularizer/mulMul5rsp__model/conv2d_3/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_3/kernel/Regularizer/mul?
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1293644* 
_output_shapes
:
??*
dtype02;
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp?
*rsp__model/dense/kernel/Regularizer/SquareSquareArsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2,
*rsp__model/dense/kernel/Regularizer/Square?
)rsp__model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)rsp__model/dense/kernel/Regularizer/Const?
'rsp__model/dense/kernel/Regularizer/SumSum.rsp__model/dense/kernel/Regularizer/Square:y:02rsp__model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/Sum?
)rsp__model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2+
)rsp__model/dense/kernel/Regularizer/mul/x?
'rsp__model/dense/kernel/Regularizer/mulMul2rsp__model/dense/kernel/Regularizer/mul/x:output:00rsp__model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'rsp__model/dense/kernel/Regularizer/mul?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall;^rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=^rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:^rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????@@::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2x
:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp:rsp__model/conv2d/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2|
<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2v
9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp9rsp__model/dense/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????@@

_user_specified_namex
?
~
)__inference_dense_1_layer_call_fn_1294482

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_12934952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?5
?
#__inference__traced_restore_1294642
file_prefix-
)assignvariableop_rsp__model_conv2d_kernel-
)assignvariableop_1_rsp__model_conv2d_bias1
-assignvariableop_2_rsp__model_conv2d_1_kernel/
+assignvariableop_3_rsp__model_conv2d_1_bias1
-assignvariableop_4_rsp__model_conv2d_2_kernel/
+assignvariableop_5_rsp__model_conv2d_2_bias1
-assignvariableop_6_rsp__model_conv2d_3_kernel/
+assignvariableop_7_rsp__model_conv2d_3_bias.
*assignvariableop_8_rsp__model_dense_kernel,
(assignvariableop_9_rsp__model_dense_bias1
-assignvariableop_10_rsp__model_dense_1_kernel/
+assignvariableop_11_rsp__model_dense_1_bias
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_rsp__model_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_rsp__model_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_rsp__model_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_rsp__model_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp-assignvariableop_4_rsp__model_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_rsp__model_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp-assignvariableop_6_rsp__model_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_rsp__model_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_rsp__model_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_rsp__model_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_rsp__model_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_rsp__model_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
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
?
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1294378

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02>
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
-rsp__model/conv2d_2/kernel/Regularizer/SquareSquareDrsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@?2/
-rsp__model/conv2d_2/kernel/Regularizer/Square?
,rsp__model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,rsp__model/conv2d_2/kernel/Regularizer/Const?
*rsp__model/conv2d_2/kernel/Regularizer/SumSum1rsp__model/conv2d_2/kernel/Regularizer/Square:y:05rsp__model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/Sum?
,rsp__model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,rsp__model/conv2d_2/kernel/Regularizer/mul/x?
*rsp__model/conv2d_2/kernel/Regularizer/mulMul5rsp__model/conv2d_2/kernel/Regularizer/mul/x:output:03rsp__model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*rsp__model/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<rsp__model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????@@<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
dropout
maxpool
	conv1
	conv2
	conv3
	conv4
flatten
d1
	d2

regularization_losses
trainable_variables
	variables
	keras_api

signatures
q_default_save_signature
*r&call_and_return_all_conditional_losses
s__call__"?
_tf_keras_model?{"class_name": "RSP_Model", "name": "rsp__model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "RSP_Model"}}
?
regularization_losses
trainable_variables
	variables
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
regularization_losses
trainable_variables
	variables
	keras_api
*v&call_and_return_all_conditional_losses
w__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*x&call_and_return_all_conditional_losses
y__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64, 64, 3]}}
?


kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
*z&call_and_return_all_conditional_losses
{__call__"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 31, 31, 64]}}
?


#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
*|&call_and_return_all_conditional_losses
}__call__"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 14, 14, 64]}}
?


)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
*~&call_and_return_all_conditional_losses
__call__"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 6, 6, 128]}}
?
/regularization_losses
0trainable_variables
1	variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 512]}}
?

9kernel
:bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 256]}}
H
?0
?1
?2
?3
?4"
trackable_list_wrapper
v
0
1
2
3
#4
$5
)6
*7
38
49
910
:11"
trackable_list_wrapper
v
0
1
2
3
#4
$5
)6
*7
38
49
910
:11"
trackable_list_wrapper
?

?layers

regularization_losses
@non_trainable_variables
Alayer_regularization_losses
trainable_variables
Blayer_metrics
Cmetrics
	variables
s__call__
q_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Dlayers
regularization_losses
Enon_trainable_variables
Flayer_regularization_losses
trainable_variables
Glayer_metrics
Hmetrics
	variables
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Ilayers
regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses
trainable_variables
Llayer_metrics
Mmetrics
	variables
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
2:0@2rsp__model/conv2d/kernel
$:"@2rsp__model/conv2d/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Nlayers
regularization_losses
Onon_trainable_variables
Player_regularization_losses
trainable_variables
Qlayer_metrics
Rmetrics
	variables
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
4:2@@2rsp__model/conv2d_1/kernel
&:$@2rsp__model/conv2d_1/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Slayers
regularization_losses
Tnon_trainable_variables
Ulayer_regularization_losses
 trainable_variables
Vlayer_metrics
Wmetrics
!	variables
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
5:3@?2rsp__model/conv2d_2/kernel
':%?2rsp__model/conv2d_2/bias
(
?0"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?

Xlayers
%regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses
&trainable_variables
[layer_metrics
\metrics
'	variables
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
6:4??2rsp__model/conv2d_3/kernel
':%?2rsp__model/conv2d_3/bias
(
?0"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?

]layers
+regularization_losses
^non_trainable_variables
_layer_regularization_losses
,trainable_variables
`layer_metrics
ametrics
-	variables
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

blayers
/regularization_losses
cnon_trainable_variables
dlayer_regularization_losses
0trainable_variables
elayer_metrics
fmetrics
1	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)
??2rsp__model/dense/kernel
$:"?2rsp__model/dense/bias
(
?0"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?

glayers
5regularization_losses
hnon_trainable_variables
ilayer_regularization_losses
6trainable_variables
jlayer_metrics
kmetrics
7	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*	?2rsp__model/dense_1/kernel
%:#2rsp__model/dense_1/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?

llayers
;regularization_losses
mnon_trainable_variables
nlayer_regularization_losses
<trainable_variables
olayer_metrics
pmetrics
=	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
_
0
1
2
3
4
5
6
7
	8"
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
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
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
(
?0"
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
?2?
"__inference__wrapped_model_1293254?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@@
?2?
G__inference_rsp__model_layer_call_and_return_conditional_losses_1293892
G__inference_rsp__model_layer_call_and_return_conditional_losses_1293975
G__inference_rsp__model_layer_call_and_return_conditional_losses_1294123
G__inference_rsp__model_layer_call_and_return_conditional_losses_1294206?
???
FullArgSpec-
args%?"
jself
jx
jtrain

jtraining
varargs
 
varkw
 
defaults?
p
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_rsp__model_layer_call_fn_1294033
,__inference_rsp__model_layer_call_fn_1294004
,__inference_rsp__model_layer_call_fn_1294235
,__inference_rsp__model_layer_call_fn_1294264?
???
FullArgSpec-
args%?"
jself
jx
jtrain

jtraining
varargs
 
varkw
 
defaults?
p
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_layer_call_and_return_conditional_losses_1294281
D__inference_dropout_layer_call_and_return_conditional_losses_1294276?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_layer_call_fn_1294291
)__inference_dropout_layer_call_fn_1294286?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1293261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_layer_call_fn_1293267?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_conv2d_layer_call_and_return_conditional_losses_1294314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_layer_call_fn_1294323?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1294346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_1_layer_call_fn_1294355?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1294378?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_2_layer_call_fn_1294387?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1294410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_3_layer_call_fn_1294419?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_layer_call_and_return_conditional_losses_1294425?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_layer_call_fn_1294430?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_1294453?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_layer_call_fn_1294462?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_1294473?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_1294482?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_1294493?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_1294504?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_1294515?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_1294526?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_1294537?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
%__inference_signature_wrapper_1293802input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_1293254}#$)*349:8?5
.?+
)?&
input_1?????????@@
? "3?0
.
output_1"?
output_1??????????
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1294346l7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_1_layer_call_fn_1294355_7?4
-?*
(?%
inputs?????????@
? " ??????????@?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1294378m#$7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_2_layer_call_fn_1294387`#$7?4
-?*
(?%
inputs?????????@
? "!????????????
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1294410n)*8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_3_layer_call_fn_1294419a)*8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_conv2d_layer_call_and_return_conditional_losses_1294314l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????>>@
? ?
(__inference_conv2d_layer_call_fn_1294323_7?4
-?*
(?%
inputs?????????@@
? " ??????????>>@?
D__inference_dense_1_layer_call_and_return_conditional_losses_1294473]9:0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_1_layer_call_fn_1294482P9:0?-
&?#
!?
inputs??????????
? "???????????
B__inference_dense_layer_call_and_return_conditional_losses_1294453^340?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_layer_call_fn_1294462Q340?-
&?#
!?
inputs??????????
? "????????????
D__inference_dropout_layer_call_and_return_conditional_losses_1294276^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
D__inference_dropout_layer_call_and_return_conditional_losses_1294281^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ~
)__inference_dropout_layer_call_fn_1294286Q4?1
*?'
!?
inputs??????????
p
? "???????????~
)__inference_dropout_layer_call_fn_1294291Q4?1
*?'
!?
inputs??????????
p 
? "????????????
D__inference_flatten_layer_call_and_return_conditional_losses_1294425b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
)__inference_flatten_layer_call_fn_1294430U8?5
.?+
)?&
inputs??????????
? "???????????<
__inference_loss_fn_0_1294493?

? 
? "? <
__inference_loss_fn_1_1294504?

? 
? "? <
__inference_loss_fn_2_1294515#?

? 
? "? <
__inference_loss_fn_3_1294526)?

? 
? "? <
__inference_loss_fn_4_12945373?

? 
? "? ?
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1293261?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_layer_call_fn_1293267?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_rsp__model_layer_call_and_return_conditional_losses_1293892w#$)*349:@?=
6?3
)?&
input_1?????????@@
p
p
? "%?"
?
0?????????
? ?
G__inference_rsp__model_layer_call_and_return_conditional_losses_1293975w#$)*349:@?=
6?3
)?&
input_1?????????@@
p
p 
? "%?"
?
0?????????
? ?
G__inference_rsp__model_layer_call_and_return_conditional_losses_1294123q#$)*349::?7
0?-
#? 
x?????????@@
p
p
? "%?"
?
0?????????
? ?
G__inference_rsp__model_layer_call_and_return_conditional_losses_1294206q#$)*349::?7
0?-
#? 
x?????????@@
p
p 
? "%?"
?
0?????????
? ?
,__inference_rsp__model_layer_call_fn_1294004j#$)*349:@?=
6?3
)?&
input_1?????????@@
p
p
? "???????????
,__inference_rsp__model_layer_call_fn_1294033j#$)*349:@?=
6?3
)?&
input_1?????????@@
p
p 
? "???????????
,__inference_rsp__model_layer_call_fn_1294235d#$)*349::?7
0?-
#? 
x?????????@@
p
p
? "???????????
,__inference_rsp__model_layer_call_fn_1294264d#$)*349::?7
0?-
#? 
x?????????@@
p
p 
? "???????????
%__inference_signature_wrapper_1293802?#$)*349:C?@
? 
9?6
4
input_1)?&
input_1?????????@@"3?0
.
output_1"?
output_1?????????