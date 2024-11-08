try:
    from setuptools import setup, find_namespace_packages
except ImportError:
    raise ImportError(
        'Some errors occurred when importing "setuptools". Maybe you should '
        'upgrade "setuptools" using \n\n"pip install -U setuptools"\n\n. When "setuptools" '
        'is updated successfully, try to install this package again :-).')

#script_dir = sys.path[0]
info_dict = {}
exec(open('deepwmh/pkginfo.py').read(), info_dict)

setup(
    name='deepwmh',
    packages=find_namespace_packages(include=["deepwmh", "deepwmh.*"]),
    version=info_dict['__version__'],
    
    description=
    'DeepWMH: An accurate annotation-free white matter hyperintensity lesion segmentation algorithm '
    'for medical image analysis based on multi-atlas image registration and negative log-likelihood.',
    
    author=info_dict['__author__'],
    author_email='chenghao1652@126.com, 3120215380@bit.edu.cn',
    
    # need Python > 3.5, because os.mkdir(...) is not thread-safe before this version
    # Python 3.7 just works fine.

    #
    # NOTE: these package settings are only tested on Ubuntu-16.04
    #
    python_requires='>=3.7.1', 
    install_requires=[
        'numpy>=1.20.3',
        'nibabel==3.2.1',
        'scikit-image==0.18.1',
        'scipy==1.10.0',
        'matplotlib==3.4.2',
        'psutil==5.8.0',
        'imageio==2.13.2',
        'openpyxl==3.0.9', # read xlsx
        'xlsxwriter==1.4.3', # write xlsx
        #'pingouin==0.5.1', # for correlation analysis
        #'statsmodels==0.13.2',
        'reportlab==3.6.9',
        'svglib==1.2.1'
    ],
    entry_points={
        'console_scripts':[
            'antsGroupRegistration = deepwmh.external_tools.ANTs_group_registration:main',
            'dcm2niix_py = deepwmh.external_tools.dcm2niix:main',
            'DeepWMH_train = deepwmh.main.train:main',
            'DeepWMH_predict = deepwmh.main.predict:main',
            'DeepWMH_install = deepwmh.main.install_model:main'
        ]
    },
    keywords=[
        'image segmentation', 'lesion segmentation' , 'unsupervised learning',
        'annotation-free', 'annotation-free lesion segmentation', 'deepwmh',
        'medical image segmentation', 'unsupervised lesion segmentation', 
        'anomaly detection', 'NLL', 'negative log-likelihood',
        'white matter lesions', 'WMH', 'white matter hyperintensities'
    ]
)
