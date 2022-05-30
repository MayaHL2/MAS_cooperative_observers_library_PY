import bpy

cube = bpy.ops.mesh.primitive_cube_add()
bpy.ops.transform.resize(value=(1, 1, 0.2), 
                        orient_type='GLOBAL', 
                        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                        orient_matrix_type='GLOBAL', 
                        constraint_axis=(Falase, False, True),
                        mirror=False, 
                        use_proportional_edit=False, 
                        proportional_edit_falloff='SMOOTH', 
                        proportional_size=1, 
                        use_proportional_connected=False, 
                        use_proportional_projected=False, 
                        release_confirm=True)

a
#so = bpy.context.active_object

