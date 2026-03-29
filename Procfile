python -c "
with open('Procfile', 'w') as f:
    f.write('web: python -m uvicorn main:app --host 0.0.0.0 --port \$PORT\n')
print('Done')
"