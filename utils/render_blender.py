# -*- coding: utf-8 -*-
# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Tested with Blender 2.9
#
# Example:
# blender --background --python render_blender.py -- --obj D:\workspace\project\image-relighting\data\obj\ironman\IronMan.obj
# 如果用HHuman数据集的话，角度要改成-200
import argparse, sys, os, math, re
import bpy
from glob import glob

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=1,
                    help='number of views to be rendered')
parser.add_argument('--obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./output/bld/',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='16',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--resolution', type=int, default=1000,
                    help='Resolution of the images.')
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
parser.add_argument('--render_mask', type=bool, default=True,
                    help='Render object mask if set to True.')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = args.color_depth # ('8', '16')
render.image_settings.file_format = args.format # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
render.film_transparent = True #设置背景是否透明

scene.use_nodes = True
scene.view_layers["View Layer"].use_pass_normal = True
scene.view_layers["View Layer"].use_pass_diffuse_color = True
scene.view_layers["View Layer"].use_pass_object_index = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')
print("Available lights:", bpy.data.lights.keys())
# Create depth output nodes
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
depth_file_output.base_path = ''
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = args.format
depth_file_output.format.color_depth = args.color_depth
if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
    depth_file_output.format.color_mode = "BW"

    # Remap as other types can not represent the full range of depth.
    map = nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.offset = [-0.7]
    map.size = [args.depth_scale]
    map.use_min = True
    map.min = [0]
    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])


# Create normal output nodes
scale_node = nodes.new(type="CompositorNodeMixRGB")
scale_node.blend_type = 'MULTIPLY'
scale_node.use_alpha = True
scale_node.inputs[2].default_value = (1, 1, 1, 1)
links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

bias_node = nodes.new(type="CompositorNodeMixRGB")
bias_node.blend_type = 'MULTIPLY'
bias_node.use_alpha = True
bias_node.inputs[2].default_value = (1, 1, 1, 1)
links.new(scale_node.outputs[0], bias_node.inputs[1])

normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
normal_file_output.base_path = ''
normal_file_output.file_slots[0].use_node_format = True
normal_file_output.format.file_format = args.format


links.new(bias_node.outputs[0], normal_file_output.inputs[0])

# Create albedo output nodes
alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
albedo_file_output.base_path = ''
albedo_file_output.file_slots[0].use_node_format = True
albedo_file_output.format.file_format = args.format
albedo_file_output.format.color_mode = 'RGBA'
albedo_file_output.format.color_depth = args.color_depth
links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

# Create id map output nodes
id_file_output = nodes.new(type="CompositorNodeOutputFile")
id_file_output.label = 'ID Output'
id_file_output.base_path = ''
id_file_output.file_slots[0].use_node_format = True
id_file_output.format.file_format = args.format
id_file_output.format.color_depth = args.color_depth

if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
else:
    id_file_output.format.color_mode = 'BW'

    divide_node = nodes.new(type='CompositorNodeMath')
    divide_node.operation = 'DIVIDE'
    divide_node.use_clamp = False
    divide_node.inputs[1].default_value = 2**int(args.color_depth)

    links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
    links.new(divide_node.outputs[0], id_file_output.inputs[0])
# Create id map output nodes
if args.render_mask:
    mask_file_output = nodes.new(type="CompositorNodeOutputFile")
    mask_file_output.label = 'Mask Output'
    mask_file_output.base_path = ''
    mask_file_output.file_slots[0].use_node_format = True
    mask_file_output.format.file_format = args.format
    mask_file_output.format.color_depth = args.color_depth

    if args.format == 'OPEN_EXR':
        links.new(render_layers.outputs['IndexOB'], mask_file_output.inputs[0])
    else:
        mask_file_output.format.color_mode = 'BW'

        divide_node_mask = nodes.new(type='CompositorNodeMath')
        divide_node_mask.operation = 'DIVIDE'
        divide_node_mask.use_clamp = False
        divide_node_mask.inputs[1].default_value = 2**int(args.color_depth)

        links.new(render_layers.outputs['IndexOB'], divide_node_mask.inputs[0])
        links.new(divide_node_mask.outputs[0], mask_file_output.inputs[0])


# Delete default cube
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')

bpy.ops.import_scene.obj(filepath=args.obj)

obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
# rotate obj
obj.location = (0, 0, 0)
# 如果是HUUMAN数据集，则设置为-200 。 其他的0
rotation_in_degrees = -200
rotation_in_radians = math.radians(rotation_in_degrees)
obj.rotation_euler = (rotation_in_radians, 0, 0)
# obj.rotation_euler = (-180, 0, 0)
# get size of model and size it
initial_scale = obj.dimensions
max_index = initial_scale.to_tuple().index(max(initial_scale))
# print(max_index)
# print("*******************************************************************************************************")
scale_rate = 7.0 / initial_scale[max_index]
bpy.ops.transform.resize(value=(scale_rate, scale_rate, scale_rate))
bpy.ops.object.transform_apply(scale=True)

#load hdr map
# hdr_path = r"C:\Users\Administrator\Pictures\HDR\night_2k.hdr"

# Possibly disable specular shading
for slot in obj.material_slots:
    node = slot.material.node_tree.nodes['Principled BSDF']
    node.inputs['Specular'].default_value = 0.05

if args.scale != 1:
    bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
    bpy.ops.object.transform_apply(scale=True)
if args.remove_doubles:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode='OBJECT')
if args.edge_split:
    bpy.ops.object.modifier_add(type='EDGE_SPLIT')
    context.object.modifiers["EdgeSplit"].split_angle = 1.32645
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

# Set objekt IDs
obj.pass_index = 1

# Make light just directional, disable shadows.
light = bpy.data.lights['Light']
light.type = 'SUN'
light.use_shadow = False
# Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 8.0

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.light_add(type='SUN')
light2 = bpy.context.active_object.data
light2.use_shadow = False
light2.specular_factor = 1.0
light2.energy = 0.15
bpy.context.active_object.rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.context.active_object.rotation_euler[0] += 180

# Place camera
cam = scene.objects['Camera']
cam.location = (0, 0, 10)
cam.rotation_euler = (math.radians(90), 0, 0)
#
cam.data.lens = 35
cam.data.sensor_width = 32
#
#
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'

# model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
# fp = os.path.join(os.path.abspath(args.output_folder), model_identifier, model_identifier)
model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
fp = os.path.abspath(os.path.join(args.output_folder, model_identifier))

for i in range(-30,30,10):

    for j in range(-30,30,10):

        obj.rotation_euler[0] = math.radians(j)
        print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

        # render_file_path = fp + '_r_{0:03d}'.format(int(i * stepsize))
        id_path = os.path.join(fp,'image')
        normal_path = os.path.join(fp,'normal')
        albedo_path = os.path.join(fp,'albedo')
        mask_path = os.path.join(fp, 'mask')
        os.makedirs(id_path, exist_ok=True)
        os.makedirs(normal_path, exist_ok=True)
        os.makedirs(albedo_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        id_render_path = os.path.join(id_path, 'x_{0:04d}_y_{1:04d}'.format(int(i * 1.0), j))
        normal_render_path = os.path.join(normal_path, 'x_{0:04d}_y_{1:04d}_normal'.format(int(i * 1.0), j))
        alebdo_render_path = os.path.join(albedo_path, 'x_{0:04d}_y_{1:04d}_albedo'.format(int(i * 1.0), j))

        render_file_path = os.path.join(id_path, 'x_{0:04d}_y_{1:04d}_image'.format(int(i * 1.0), j))

        if args.render_mask:
            mask_render_path = os.path.join(mask_path, 'x_{0:04d}_y_{1:04d}_mask'.format(int(i * 1.0), j))
            mask_file_output.file_slots[0].path = mask_render_path

        # render_file_path = id_render_path + '_x_{0:04d}_y_{1:04d}'.format(int(i * 1.0), j)
        # scene.render.filepath = render_file_path
        scene.render.filepath = id_render_path

        # depth_file_output.file_slots[0].path = render_file_path + "_depth"
        normal_file_output.file_slots[0].path = normal_render_path
        albedo_file_output.file_slots[0].path = alebdo_render_path
        id_file_output.file_slots[0].path = id_render_path

        bpy.ops.render.render(write_still=True)  # render still

    # cam_empty.rotation_euler[1] += math.radians(stepsize)
    obj.rotation_euler[1] = math.radians(i)
    obj.rotation_euler[0] = math.radians(0)
# For debugging the workflow
#bpy.ops.wm.save_as_mainfile(filepath='debug.blend')
