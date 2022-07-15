# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: feature_selection.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='feature_selection.proto',
  package='featureselection',
  syntax='proto3',
  serialized_options=b'\n!io.grpc.examples.featureselectionB\025FeatureSelectionProtoP\001\242\002\003FSP',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x17\x66\x65\x61ture_selection.proto\x12\x10\x66\x65\x61tureselection\"\x14\n\x04\x46lag\x12\x0c\n\x04\x66lag\x18\x01 \x01(\x08\"4\n\x15SamplingRowsAndTarget\x12\x0b\n\x03row\x18\x01 \x03(\x05\x12\x0e\n\x06target\x18\x02 \x03(\x02\"\x14\n\x05\x41rray\x12\x0b\n\x03num\x18\x01 \x03(\x05\"\x19\n\nFloatArray\x12\x0b\n\x03num\x18\x01 \x03(\x02\"M\n\rCrossFeatures\x12\x14\n\x0c\x66\x65\x61ture_name\x18\x01 \x03(\t\x12&\n\x05\x61rray\x18\x02 \x03(\x0b\x32\x17.featureselection.Array\"U\n\x10SelectedFeatures\x12\x14\n\x0c\x66\x65\x61ture_name\x18\x01 \x03(\t\x12+\n\x05\x61rray\x18\x02 \x03(\x0b\x32\x1c.featureselection.FloatArray\"#\n\x0bSelectedNum\x12\x14\n\x0cselected_num\x18\x01 \x01(\x05\"\x19\n\x06\x44\x65lNum\x12\x0f\n\x07\x64\x65l_num\x18\x01 \x01(\x05\"\x1e\n\rFeatureScores\x12\r\n\x05score\x18\x01 \x03(\x02\x32\xf4\x03\n\x03MFS\x12^\n\x19SendSamplingRowsAndTarget\x12\'.featureselection.SamplingRowsAndTarget\x1a\x16.featureselection.Flag\"\x00\x12W\n\x11SendCrossFeatures\x12\x1f.featureselection.CrossFeatures\x1a\x1f.featureselection.FeatureScores\"\x00\x12\x42\n\x05GetMI\x12\x16.featureselection.Flag\x1a\x1f.featureselection.FeatureScores\"\x00\x12T\n\x10GetCrossFeatures\x12\x1d.featureselection.SelectedNum\x1a\x1f.featureselection.CrossFeatures\"\x00\x12\x45\n\x0fSendDelFeatures\x12\x18.featureselection.DelNum\x1a\x16.featureselection.Flag\"\x00\x12S\n\x13GetSelectedFeatures\x12\x16.featureselection.Flag\x1a\".featureselection.SelectedFeatures\"\x00\x42\x42\n!io.grpc.examples.featureselectionB\x15\x46\x65\x61tureSelectionProtoP\x01\xa2\x02\x03\x46SPb\x06proto3'
)




_FLAG = _descriptor.Descriptor(
  name='Flag',
  full_name='featureselection.Flag',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='flag', full_name='featureselection.Flag.flag', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=45,
  serialized_end=65,
)


_SAMPLINGROWSANDTARGET = _descriptor.Descriptor(
  name='SamplingRowsAndTarget',
  full_name='featureselection.SamplingRowsAndTarget',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='row', full_name='featureselection.SamplingRowsAndTarget.row', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='target', full_name='featureselection.SamplingRowsAndTarget.target', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=67,
  serialized_end=119,
)


_ARRAY = _descriptor.Descriptor(
  name='Array',
  full_name='featureselection.Array',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='num', full_name='featureselection.Array.num', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=121,
  serialized_end=141,
)


_FLOATARRAY = _descriptor.Descriptor(
  name='FloatArray',
  full_name='featureselection.FloatArray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='num', full_name='featureselection.FloatArray.num', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=143,
  serialized_end=168,
)


_CROSSFEATURES = _descriptor.Descriptor(
  name='CrossFeatures',
  full_name='featureselection.CrossFeatures',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature_name', full_name='featureselection.CrossFeatures.feature_name', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='array', full_name='featureselection.CrossFeatures.array', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=170,
  serialized_end=247,
)


_SELECTEDFEATURES = _descriptor.Descriptor(
  name='SelectedFeatures',
  full_name='featureselection.SelectedFeatures',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature_name', full_name='featureselection.SelectedFeatures.feature_name', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='array', full_name='featureselection.SelectedFeatures.array', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=249,
  serialized_end=334,
)


_SELECTEDNUM = _descriptor.Descriptor(
  name='SelectedNum',
  full_name='featureselection.SelectedNum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='selected_num', full_name='featureselection.SelectedNum.selected_num', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=336,
  serialized_end=371,
)


_DELNUM = _descriptor.Descriptor(
  name='DelNum',
  full_name='featureselection.DelNum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='del_num', full_name='featureselection.DelNum.del_num', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=373,
  serialized_end=398,
)


_FEATURESCORES = _descriptor.Descriptor(
  name='FeatureScores',
  full_name='featureselection.FeatureScores',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='score', full_name='featureselection.FeatureScores.score', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=400,
  serialized_end=430,
)

_CROSSFEATURES.fields_by_name['array'].message_type = _ARRAY
_SELECTEDFEATURES.fields_by_name['array'].message_type = _FLOATARRAY
DESCRIPTOR.message_types_by_name['Flag'] = _FLAG
DESCRIPTOR.message_types_by_name['SamplingRowsAndTarget'] = _SAMPLINGROWSANDTARGET
DESCRIPTOR.message_types_by_name['Array'] = _ARRAY
DESCRIPTOR.message_types_by_name['FloatArray'] = _FLOATARRAY
DESCRIPTOR.message_types_by_name['CrossFeatures'] = _CROSSFEATURES
DESCRIPTOR.message_types_by_name['SelectedFeatures'] = _SELECTEDFEATURES
DESCRIPTOR.message_types_by_name['SelectedNum'] = _SELECTEDNUM
DESCRIPTOR.message_types_by_name['DelNum'] = _DELNUM
DESCRIPTOR.message_types_by_name['FeatureScores'] = _FEATURESCORES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Flag = _reflection.GeneratedProtocolMessageType('Flag', (_message.Message,), {
  'DESCRIPTOR' : _FLAG,
  '__module__' : 'feature_selection_pb2'
  # @@protoc_insertion_point(class_scope:featureselection.Flag)
  })
_sym_db.RegisterMessage(Flag)

SamplingRowsAndTarget = _reflection.GeneratedProtocolMessageType('SamplingRowsAndTarget', (_message.Message,), {
  'DESCRIPTOR' : _SAMPLINGROWSANDTARGET,
  '__module__' : 'feature_selection_pb2'
  # @@protoc_insertion_point(class_scope:featureselection.SamplingRowsAndTarget)
  })
_sym_db.RegisterMessage(SamplingRowsAndTarget)

Array = _reflection.GeneratedProtocolMessageType('Array', (_message.Message,), {
  'DESCRIPTOR' : _ARRAY,
  '__module__' : 'feature_selection_pb2'
  # @@protoc_insertion_point(class_scope:featureselection.Array)
  })
_sym_db.RegisterMessage(Array)

FloatArray = _reflection.GeneratedProtocolMessageType('FloatArray', (_message.Message,), {
  'DESCRIPTOR' : _FLOATARRAY,
  '__module__' : 'feature_selection_pb2'
  # @@protoc_insertion_point(class_scope:featureselection.FloatArray)
  })
_sym_db.RegisterMessage(FloatArray)

CrossFeatures = _reflection.GeneratedProtocolMessageType('CrossFeatures', (_message.Message,), {
  'DESCRIPTOR' : _CROSSFEATURES,
  '__module__' : 'feature_selection_pb2'
  # @@protoc_insertion_point(class_scope:featureselection.CrossFeatures)
  })
_sym_db.RegisterMessage(CrossFeatures)

SelectedFeatures = _reflection.GeneratedProtocolMessageType('SelectedFeatures', (_message.Message,), {
  'DESCRIPTOR' : _SELECTEDFEATURES,
  '__module__' : 'feature_selection_pb2'
  # @@protoc_insertion_point(class_scope:featureselection.SelectedFeatures)
  })
_sym_db.RegisterMessage(SelectedFeatures)

SelectedNum = _reflection.GeneratedProtocolMessageType('SelectedNum', (_message.Message,), {
  'DESCRIPTOR' : _SELECTEDNUM,
  '__module__' : 'feature_selection_pb2'
  # @@protoc_insertion_point(class_scope:featureselection.SelectedNum)
  })
_sym_db.RegisterMessage(SelectedNum)

DelNum = _reflection.GeneratedProtocolMessageType('DelNum', (_message.Message,), {
  'DESCRIPTOR' : _DELNUM,
  '__module__' : 'feature_selection_pb2'
  # @@protoc_insertion_point(class_scope:featureselection.DelNum)
  })
_sym_db.RegisterMessage(DelNum)

FeatureScores = _reflection.GeneratedProtocolMessageType('FeatureScores', (_message.Message,), {
  'DESCRIPTOR' : _FEATURESCORES,
  '__module__' : 'feature_selection_pb2'
  # @@protoc_insertion_point(class_scope:featureselection.FeatureScores)
  })
_sym_db.RegisterMessage(FeatureScores)


DESCRIPTOR._options = None

_MFS = _descriptor.ServiceDescriptor(
  name='MFS',
  full_name='featureselection.MFS',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=433,
  serialized_end=933,
  methods=[
  _descriptor.MethodDescriptor(
    name='SendSamplingRowsAndTarget',
    full_name='featureselection.MFS.SendSamplingRowsAndTarget',
    index=0,
    containing_service=None,
    input_type=_SAMPLINGROWSANDTARGET,
    output_type=_FLAG,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SendCrossFeatures',
    full_name='featureselection.MFS.SendCrossFeatures',
    index=1,
    containing_service=None,
    input_type=_CROSSFEATURES,
    output_type=_FEATURESCORES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetMI',
    full_name='featureselection.MFS.GetMI',
    index=2,
    containing_service=None,
    input_type=_FLAG,
    output_type=_FEATURESCORES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetCrossFeatures',
    full_name='featureselection.MFS.GetCrossFeatures',
    index=3,
    containing_service=None,
    input_type=_SELECTEDNUM,
    output_type=_CROSSFEATURES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SendDelFeatures',
    full_name='featureselection.MFS.SendDelFeatures',
    index=4,
    containing_service=None,
    input_type=_DELNUM,
    output_type=_FLAG,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetSelectedFeatures',
    full_name='featureselection.MFS.GetSelectedFeatures',
    index=5,
    containing_service=None,
    input_type=_FLAG,
    output_type=_SELECTEDFEATURES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_MFS)

DESCRIPTOR.services_by_name['MFS'] = _MFS

# @@protoc_insertion_point(module_scope)