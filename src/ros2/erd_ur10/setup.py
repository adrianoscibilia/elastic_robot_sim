from setuptools import find_packages, setup

package_name = "erd_ur10"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/ur10.launch.py"]),
        (f"share/{package_name}/config", ["config/controllers.yaml"]),
    ],
    scripts=["scripts/start_ursim.sh"],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="adriano",
    maintainer_email="adriano.scibilia@cnr.it",
    description="UR10 CB3 launch + RTDE sidecar logger (RR_01 S3.4).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rtde_logger = erd_ur10.rtde_logger:main",
        ],
    },
)
