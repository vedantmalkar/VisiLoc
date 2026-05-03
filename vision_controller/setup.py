from setuptools import find_packages, setup

package_name = 'vision_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vedant Malkar',
    maintainer_email='vedantitsme@gmail.com',
    description='ROS 2 perception package for VisiLoc. Subscribes to a camera image stream, detects ArUco markers, computes the robot pose with respect to the world frame, and exposes a Tkinter visualiser for the estimated coordinates.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "coordinate_finder = vision_controller.coordinate_finder:main",
            "coordinate_visualizer = vision_controller.coordinate_visualisation:main",
        ],
    },
)
