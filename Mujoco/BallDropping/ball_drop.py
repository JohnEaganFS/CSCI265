import mujoco

xml = """
<mujoco>
    <!-- Want to have a sphere -->
    <worldbody>
        <!-- Include floor -->
        <geom type="box" size="1 1 0.1" pos="0 0 0" rgba="0 1 0 1"/>
        <body name="sphere" pos="0 0 1">
            <joint type="free" name="sphere_joint"/>
            <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""


# Model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

mujoco.viewer.launch()
