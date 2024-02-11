# [AIJansou (mjai.app)](https://mjai.app/) Support

AIJansou (mjai.app) is a Riichi Mahjong AI competition platform. This directory contains a set of tools for preparing kanachan models to participate in AIJansou.

## `build.sh`

`build.sh` is a script that converts the model trained with kanachan into a `.zip` file for submission to AIJansou. First, prepare a model file (a file with the `.kanachan` extension). Next, with the top-level directory of the working tree of this repository as the current directory, execute the following command:

```
$ mjai.app/build.sh PATH_TO_MODEL_FILE
```

That's it! Now, just upload the `mjai-app.zip` file generated in the top-level directory to AIJansou and enjoy!

## `test.sh` (For Developpers Only)

This directory also contains the `test.sh` script for testing the functionality of the generated `mjai-app.zip` file before actually submitting it to AIJansou. To execute this script, with the top-level directory of working tree of this repository as the current directory, run the following command:

```
$ mjai.app/test.sh
```

This script performs repeated self-matches using four replicas of the model bundled in `mjai-app.zip` to check for any errors. Once this script starts running, it will not stop unless forcibly halted, whether by repeatedly pressing the `^C` key sequence or sending a `KILL` signal. Logs for each self-match are output to a directory named `logs.YYYY-MM-DD-hh-mm-ss`, where `YYYY-MM-DD-hh-mm-ss` is the start time of each self-match. Log directories that are determined to be clearly error-free are automatically deleted.
