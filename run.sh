#!/bin/sh
if [ $# -eq 1 ]
then
    python -c "from main import $1;$1()";
elif [ $# -eq 2 ]
then
    python -c "from main import $1;$1('$2')";
else
    python -c "from main import $1;$1('$2', '$3')";
fi
