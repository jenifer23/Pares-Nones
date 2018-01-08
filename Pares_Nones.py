
#Librerias 
from random import choice
from sklearn.neural_network import MLPClassifier
from bokeh.plotting import figure, show
from bokeh.io import push_notebook, show, output_notebook
#Se lista las opciones del juego
opciones = ["uno","dos","tres","cuatro","cinco"]
#funcion para escoger al ganador
def buscar_ganador(p1,p2):
    if p1 == p2:
        result = 1
    #1
    elif p1 == "uno" and p2 =="dos":
        result = 2
    elif p1 == "uno" and p2 =="tres":
        result = 1
    elif p1 == "uno" and p2 =="cuatro":
        result = 2
    elif p1 == "uno" and p2 =="cinco":
        result = 1
    #2
    elif p1 == "dos" and p2 =="uno":
        result = 2
    elif p1 == "dos" and p2 =="tres":
        result = 2
    elif p1 == "dos" and p2 =="cuatro":
        result = 1
    elif p1 == "dos" and p2 =="cinco":
        result = 2
    #3
    elif p1 == "tres" and p2 =="uno":
        result = 1
    elif p1 == "tres" and p2 =="dos":
        result = 2
    elif p1 == "tres" and p2 =="cuatro":
        result = 2
    elif p1 == "tres" and p2 =="cinco":
        result = 1
    #4
    elif p1 == "cuatro" and p2 =="uno":
        result = 2
    elif p1 == "cuatro" and p2 =="dos":
        result = 1
    elif p1 == "cuatro" and p2 =="tres":
        result = 2
    elif p1 == "cuatro" and p2 =="cinco":
        result = 2
    #5
    elif p1 == "cinco" and p2 =="uno":
        result = 1
    elif p1 == "cinco" and p2 =="dos":
        result = 2
    elif p1 == "cinco" and p2 =="tres":
        result = 1
    elif p1 == "cinco" and p2 =="cuatro":
        result = 2

    return result
#funcion para elegir aleatoriamente una opcion
def get_eleccion():
    return choice(opciones)
#funcion para convertir a una lista 
def str_to_list(opcion):
    if opcion == "cinco":
        res = [1,0,0,0,0]
    elif opcion == "cuatro":
        res = [0,1,0,0,0]
    elif opcion == "tres":
        res = [0,0,1,0,0]
    elif opcion == "dos":
        res = [0,0,0,1,0]
    else:
        res = [0,0,0,0,1]
    return res
#Mapeamos en la una lista
data_x = list(map(str_to_list,["uno","dos","tres","cuatro","cinco"]))
data_y = list(map(str_to_list,["dos","tres","cinco","cuatro","uno"]))

clf = MLPClassifier(verbose=False, warm_start=True)
model = clf.fit([data_x[0]], [data_y[0]])

#funcion para que aprenda
def jugar_aprender(it = 10,debug = False):
    puntuacion = {"ganadas" : 0,"perdidas" : 0}
    data_x =[]
    data_y =[]
    for i in range(it):
        jugador1 = get_eleccion()

        prediccion = model.predict_proba([str_to_list(jugador1)])[0] 
        
        if prediccion[0] >= 0.90:
            jugador2 = opciones[0]
        elif prediccion[1] >= 0.90:
            jugador2 = opciones[1]
        elif prediccion[2] >= 0.90:
            jugador2 = opciones[2]
        elif prediccion[3] >= 0.90:
            jugador2 = opciones[3]
        elif prediccion[4] >= 0.90:
            jugador2 = opciones[4]
        else:
            jugador2 = get_eleccion()
        
        if debug == True:
            print("Jugador 1: %s Jugador 2 (modelo): %s --> %s" % (jugador1, prediccion, jugador2))
        
        ganador = buscar_ganador(jugador1,jugador2)
       
        if debug==True:
            print("Comprobamos: j1 VS j2: %s" % ganador)
        if ganador==2:
            data_x.append(str_to_list(jugador1))
            data_y.append(str_to_list(jugador2))       
            puntuacion["ganadas"]+=1
        else:
            puntuacion["perdidas"]+=1
    
    return puntuacion,data_x,data_y

#Se hace una prueba para ver como funciona nuestra red y aprende la jugada
puntuacion,data_x,data_y = jugar_aprender(1, debug = True)
print(data_x)
print(data_y)
print("Puntuacion: %s %s %% " % (puntuacion,(puntuacion["ganadas"]*100/(puntuacion["ganadas"]+puntuacion["perdidas"]))))
if len(data_x):
    model = model.partial_fit(data_x,data_y)
    
#Se realiza el entrenamiento de la red
i =0
historico=[]
while True:
    i+=1
    puntuacion,data_x,data_y = jugar_aprender(1000,debug=False)
    pct = (puntuacion["ganadas"]*100/(puntuacion["ganadas"]+puntuacion["perdidas"]))
    historico.append(pct)
    print("Iteracion: %s - puntuacion: %s %s %%" % (i,puntuacion,pct))
    
    if len(data_x):
        model=model.partial_fit(data_x,data_y)
    
    if sum(historico[-1:])>52:
        break
        
#Se grafica
x = range(len(historico))
y = historico

p = figure( 
    title = "Porcentaje de aprendizaje en cada iteracion",
    x_axis_label="Iter", y_axis_label="%", width=900)

p.line(x,y,legend=None,line_width=1)
show(p)

