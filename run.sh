if [ $# -eq 1 ]
then
    python -c "from main import $1;$1()";
else
    python -c "from main import $1;$1('$2')";
fi
