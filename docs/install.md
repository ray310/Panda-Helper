---
hide:
  - navigation
  - toc
description: How to install Panda-Helper. Install pandahelper with pip or anaconda and get up and running in minutes.
---

# Installing Panda-Helper

## Installing with `pip`

```shell
pip install pandahelper
```

## Installing with `conda`
If you manage conda environments with a `.yaml` file you can add `pandahelper`
to the pip section of the .yaml as show here:
```yaml
name: my_env
channels:
    - defaults
dependencies:
    - python=<version>
    - pandas
    - pip
    - pip:
        - pandahelper
```
Then rebuild or update your conda environment.
