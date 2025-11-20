# Full.sh
#!/bin/bash

# Ejecutar el entrenamiento en segundo plano y guardar log
nohup ./nohup/run.sh > ./source/logs/salida.log 2>&1

# Ejecutar el Bot en segundo plano y guardar log
nohup ./nohup/runBot.sh > ./source/logs/salidaBot.log 2>&1

# Ejecutar el GLUe en segundo plano y guardar log
nohup ./nohup/runGLUE.sh > ./source/logs/salidaGLUE.log 2>&1
