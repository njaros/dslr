if [ -d .venv/ ]; then
    source .venv/bin/activate

else
    python3 -m venv .venv --upgrade-deps
    source .venv/bin/activate
    pip install -r requirements.txt
    cp requirements.txt .venv/requirements.txt
fi

if ! diff "requirements.txt" ".venv/requirements.txt" > /dev/null; then
    pip install -r requirements.txt
    cp requirements.txt .venv/requirements.txt
fi
