# =========================
# Importuri necesare
# =========================


import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from CTkMessagebox import CTkMessagebox
import time
import random
import math
import csv
import tkinter as tk


# =========================
# Liste pentru stocarea valorilor N, timpilor și costurilor pentru fiecare algoritm
# =========================


N = [] 

TimpiGA = []
CosturiGA = []

TimpiBCKT = []
CosturiBCKT = []

TimpiNN = []
CosturiNN = []

TimpiSA = []
CosturiSA = []


# =========================
# Variabile globale pentru istoric și indexare fișiere
# =========================


Istoric = ""
NumarComanda = 1
indiceFisier = 0
indiceFisierDiversi = 0


# =========================
# Funcție pentru afișarea graficelor (timpi și costuri)
# =========================


def ImplementareGrafic(x, y1, y2, y3, y4, y11, y22, y33, y44):
    bar_width = 0.2
    x_indexes = np.arange(len(x))

    # Primul grafic: Timpi
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(7, 3.8)

    ax1.bar(x_indexes - 1.5 * bar_width, y1, width=bar_width, color="blue")
    ax1.bar(x_indexes - 0.5 * bar_width, y2, width=bar_width, color="orange")
    ax1.bar(x_indexes + 0.5 * bar_width, y3, width=bar_width, color="green")
    ax1.bar(x_indexes + 1.5 * bar_width, y4, width=bar_width, color="purple")

    ax1.set_xlabel('N')
    ax1.set_ylabel('Timp (s)')
    ax1.set_title('Problema comis-voiajorului - Timpul de execuție')
    ax1.set_xticks(x_indexes)
    ax1.set_xticklabels(x)
    ax1.legend(["BCKT", "NN", "SA", "GA"])

    canvas1 = FigureCanvasTkAgg(fig1, master=root_tk)
    canvas1.draw()
    canvas1.get_tk_widget().place(relx=0.455, rely=0.03)

    # Al doilea grafic: Costuri
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(7, 3.8)

    ax2.bar(x_indexes - 1.5 * bar_width, y11, width=bar_width, color="blue")
    ax2.bar(x_indexes - 0.5 * bar_width, y22, width=bar_width, color="orange")
    ax2.bar(x_indexes + 0.5 * bar_width, y33, width=bar_width, color="green")
    ax2.bar(x_indexes + 1.5 * bar_width, y44, width=bar_width, color="purple")

    ax2.set_xlabel('N')
    ax2.set_ylabel('Cost')
    ax2.set_title('Problema comis-voiajorului - Costurile soluțiilor\n (Valoare mai mică = mai bună)')
    ax2.set_xticks(x_indexes)
    ax2.set_xticklabels(x)
    ax2.legend(["BCKT", "NN", "SA", "GA"])

    canvas2 = FigureCanvasTkAgg(fig2, master=root_tk)
    canvas2.draw()
    canvas2.get_tk_widget().place(relx=0.455, rely=0.52)


# =========================
# NUA
# =========================


'''
#Implementarea graficului
def ImplementareGrafic(x, y1, y2, y3, y4, y11, y22, y33, y44):

    #Primul grafic, ce contine timpii

    fig, ax= plt.subplots()
    fig.set_size_inches(7.5,4)
    ax.plot(x, y1, marker = 'o', scalex = 1, scaley = 1, color = "Blue", linestyle='-')
    ax.plot(x, y2, marker = 'o', scalex = 1, scaley = 1, color = "Orange", linestyle='-')
    ax.plot(x, y3, marker = 'o', scalex = 1, scaley = 1, color = "Green", linestyle='-')
    ax.plot(x, y4, marker = 'o', scalex = 1, scaley = 1, color = "Purple", linestyle='-')
    plt.plot(marker = 11, side='right')
    plt.xlabel('N')
    plt.ylabel('Timp (s)')
    plt.title('Problema comis-voiajorului, timpii rezolvărilor')
    plt.legend(["Rezolvare cu BCKT:", "Rezolvare cu NN", "Rezolvare cu SA", "Rezolvare cu GA"])

    canvas = FigureCanvasTkAgg(fig,master=root_tk)
    canvas.draw()
    canvas.get_tk_widget().place(relx=0.475, rely=0.03)

    #Al doilea grafic, ce contine costurile

    fig, ax= plt.subplots()
    fig.set_size_inches(7.5,4)
    ax.plot(x, y11, marker = 'o', scalex = 1, scaley = 1, color = "Blue", linestyle='-')
    ax.plot(x, y22, marker = 'o', scalex = 1, scaley = 1, color = "Orange", linestyle='-')
    ax.plot(x, y33, marker = 'o', scalex = 1, scaley = 1, color = "Green", linestyle='-')
    ax.plot(x, y44, marker = 'o', scalex = 1, scaley = 1, color = "Purple", linestyle='-')
    plt.plot(marker = 11, side='right')
    plt.xlabel('N')
    plt.ylabel('Cost')
    plt.title('Problema comis-voiajorului, costurile rezolvărilor\n (Valoarea mai mica e mai buna)')
    plt.legend(["Rezolvare cu BCKT:", "Rezolvare cu NN", "Rezolvare cu SA", "Rezolvare cu GA"])

    canvas = FigureCanvasTkAgg(fig,master=root_tk)
    canvas.draw()
    canvas.get_tk_widget().place(relx=0.475, rely=0.52)
'''


# =========================
# Pop-up pentru afișarea graficelor
# =========================


def popUpButon1(label):
    global N

    global TimpiGA
    global CosturiGA

    global TimpiBCKT
    global CosturiBCKT

    global TimpiNN
    global CosturiNN

    global TimpiSA
    global CosturiSA

    global Istoric
    global NumarComanda
    global indiceFisier
    global indiceFisierDiversi

    if N:
        # Mesaj pentru label
        text_nou = "Se încearcă afișarea graficelor...\nGraficele au fost făcute cu succes!\nValorile folosite vor fi șterse..."

        Istoric = f"{Istoric}\n{str(NumarComanda)}. {text_nou}"

        label.configure(text = Istoric)

        NumarComanda += 1

        # Afișare pop-up cu atenționare
        popup = ctk.CTkToplevel(root_tk)
        popup.title("Afișare grafice")

        popup_width = 300
        popup_height = 75
        popup.geometry(f"{popup_width}x{popup_height}")

        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()

        x = (screen_width // 2) - (popup_width // 2)
        y = (screen_height // 2) - (popup_height // 2)

        popup.geometry(f"+{x}+{y}")
    
        popup.attributes("-topmost", True)

        # Sortare valori după N, de exemplu [10, 6, 8, 15, 5] în [5, 6, 8, 10, 15]
        n = len(N)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
               if N[j] > N[j + 1]:
                    N[j], N[j + 1] = N[j + 1], N[j]
                    TimpiGA[j], TimpiGA[j + 1] = TimpiGA[j + 1], TimpiGA[j]
                    TimpiBCKT[j], TimpiBCKT[j + 1] = TimpiBCKT[j + 1], TimpiBCKT[j]
                    CosturiGA[j], CosturiGA[j+1] = CosturiGA[j+1], CosturiGA[j]
                    CosturiBCKT[j], CosturiBCKT[j+1] = CosturiBCKT[j+1], CosturiBCKT[j]
                    TimpiNN[j], TimpiNN[j+1] = TimpiNN[j+1], TimpiNN[j]
                    CosturiNN[j], CosturiNN[j+1] = CosturiNN[j+1], CosturiNN[j]
                    TimpiSA[j], TimpiSA[j + 1] = TimpiSA[j + 1], TimpiSA[j]
                    CosturiSA[j], CosturiSA[j+1] = CosturiSA[j+1], CosturiSA[j]

        # Afișare grafice
        ImplementareGrafic(N, TimpiBCKT, TimpiNN, TimpiSA, TimpiGA, CosturiBCKT, CosturiNN, CosturiSA, CosturiGA)

        # Afișare mesaj valid în label
        mesaj = ctk.CTkLabel(popup, text = "Graficele au fost făcute cu succes! Valorile folosite vor fi șterse...")
        mesaj.pack(pady = 20, padx = 20)

        # Definire funcție pentru închidere
        def inchidere():
            # Resetăm și istoricul mesajelor
            #global Istoric
            #global NumarComanda
            #global indiceFisier
            #global indiceFisierDiversi

            #Istoric = "Istoric anterior șters.\nInformații..."

            #label.configure(text = Istoric)
            #NumarComanda = 1
            #indiceFisier = 0
            #indiceFisierDiversi = 0

            popup.destroy()

        # Buton OK pentru închidere
        buton_ok = ctk.CTkButton(popup, text = "OK", command = inchidere)
        buton_ok.pack(pady = 10)

        # Ștergere valorilor pentru a nu fi afișate din nou
        N = []
        TimpiGA = []
        TimpiBCKT = []

        CosturiGA = []
        CosturiBCKT = []

        TimpiNN = []
        CosturiNN = []

        TimpiSA = []
        CosturiSA = []

        #with open("rezultate.txt", 'w') as f:
            #pass
        #with open("rezultat_tsp.csv", mode="w", newline="") as file:
            #pass

    # Dacă nu există valori în N, afișăm un mesaj de eroare
    else:
        # Mesaj pentru label
        text_nou = "Se încearcă afișarea graficelor...\nNu se pot afișa graficele. Nu există valori.\nExecută cel puțin o dată codul."

        Istoric = f"{Istoric}\n{str(NumarComanda)}. {text_nou}"

        label.configure(text = Istoric)

        NumarComanda += 1

        # Afișare pop-up dacă nu a fost rulat cel putin o dată codul
        popup = ctk.CTkToplevel(root_tk)
        popup.title("Afișare grafice")

        popup_width = 300
        popup_height = 75
        popup.geometry(f"{popup_width}x{popup_height}")

        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()

        x = (screen_width // 2) - (popup_width // 2)
        y = (screen_height // 2) - (popup_height // 2)

        popup.geometry(f"+{x}+{y}")
    
        popup.attributes("-topmost", True)


        # Afișare mesaj de eroare
        mesaj = ctk.CTkLabel(popup, text = "Nu se pot afișa graficele. Nu există valori.\nExecută cel puțin o dată codul.")
        mesaj.pack(pady = 20, padx = 20)

        # Definire funcție pentru închidere
        def inchidere():
            popup.destroy()

        # Buton OK pentru închidere
        buton_ok = ctk.CTkButton(popup, text = "OK", command = inchidere)
        buton_ok.pack(pady = 10)


# =========================
# Pop-up pentru închiderea aplicației
# =========================


def popUpButon2(label):
    text_nou = "Ești sigur că vrei să închizi?\nAplicația se va închide după apăsarea butonului..."

    global Istoric
    global NumarComanda

    Istoric = f"{Istoric}\n{str(NumarComanda)}. {text_nou}"

    label.configure(text = Istoric)

    NumarComanda += 1
     
    popup = ctk.CTkToplevel(root_tk)
    popup.title("Închidere aplicație")

    #popup.geometry("300x100")
    
    popup_width = 300
    popup_height = 75
    popup.geometry(f"{popup_width}x{popup_height}")

    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()

    x = (screen_width // 2) - (popup_width // 2)
    y = (screen_height // 2) - (popup_height // 2)

    popup.geometry(f"+{x}+{y}")
    
    popup.attributes("-topmost", True)


    # Afișare mesaj de confirmare
    mesaj = ctk.CTkLabel(popup, text = "Ești sigur că vrei să închizi? Aplicația se va închide după apăsarea butonului...")
    mesaj.pack(pady = 20, padx = 20)

    # Definire funcție pentru închidere
    def inchidere():
        popup.destroy()
        root_tk.quit()

    # Buton OK pentru închidere
    buton_ok = ctk.CTkButton(popup, text = "Închidere program", command = inchidere)
    buton_ok.pack(pady = 10)

    # Ștergere valorilor pentru a nu fi afișate din nou
    with open("rezultat_tsp.csv", mode="w", newline="") as file:
        pass
    with open("rezultat_diverse_tehnici_tsp.csv", mode="w", newline="") as file:
        pass

    #with open("rezultate.txt", 'w') as f:
            #pass


# =========================
# Pop-up pentru ștergerea istoricului
# =========================


def popUpButonIstoric(label):
    popup = ctk.CTkToplevel(root_tk)
    popup.title("Ștergere istoric")

    #popup.geometry("300x100")
    
    popup_width = 300
    popup_height = 75
    popup.geometry(f"{popup_width}x{popup_height}")

    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()

    x = (screen_width // 2) - (popup_width // 2)
    y = (screen_height // 2) - (popup_height // 2)

    popup.geometry(f"+{x}+{y}")
    
    popup.attributes("-topmost", True)

    mesaj = ctk.CTkLabel(popup, text = "Istoricul va fi șters după apăsarea butonului...")
    mesaj.pack(pady = 20, padx = 20)

    # Definire funcție pentru închidere și resetare istoric
    def inchidere():
            # Resetăm și istoricul mesajelor
            global Istoric
            global NumarComanda
            global indiceFisier
            global indiceFisierDiversi

            Istoric = "Istoric anterior șters.\nInformații..."

            label.configure(text = Istoric)
            NumarComanda = 1
            indiceFisier = 0
            indiceFisierDiversi = 0

            popup.destroy()

    # Buton OK pentru închidere și resetare istoric
    buton_ok = ctk.CTkButton(popup, text = "OK", command = inchidere)
    buton_ok.pack(pady = 10)


# =========================
# Pop-up pentru rularea diverselor tehnici GA (mutații și recombinări)
# =========================


def popUpButon6(input4, label):
    # Va fi implementat un cod nou pentru GA, ce cuprinde cele 3 tipuri de mutatii: swap, inversion, scramble
    # și cele 2 tipuri de recombinări: order crossover (OX) și partially matched crossover (PMX)
    global indiceFisierDiversi

    import time
    from random import randint, random, shuffle
    import matplotlib.pyplot as plt

    INT_MAX = 2147483647
    POP_SIZE = 10
    START = 0

    def rand_num(start, end):
        return randint(start, end - 1)
    
    def repeat(s, ch):
        return ch in s
    
    # === Mutații ===

    def swap_mutation(gnome):
        gnome = list(gnome)
        while True:
            r1 = rand_num(1, V)
            r2 = rand_num(1, V)
            if r1 != r2:
                gnome[r1], gnome[r2] = gnome[r2], gnome[r1]
                break
        return ''.join(gnome)
    
    def inversion_mutation(gnome):
        gnome = list(gnome)
        start = rand_num(1, V)
        end = rand_num(start, V)
        gnome[start:end] = gnome[start:end][::-1]
        return ''.join(gnome)
    
    def scramble_mutation(gnome):
        gnome = list(gnome)
        start = rand_num(1, V)
        end = rand_num(start, V)
        sub = gnome[start:end]
        shuffle(sub)
        gnome[start:end] = sub
        return ''.join(gnome)
    
    # === Recombinări ===

    def order_crossover(p1, p2):
        start = rand_num(1, V - 1)
        end = rand_num(start + 1, V)

        child = [''] * V
        child[start:end] = p1[start:end]

        p2_index = 1
        for i in range(1, V):
            idx = (end + i - 1) % (V - 1) + 1
            while p2_index < V and p2[p2_index] in child:
                p2_index += 1
            if p2_index < V and not child[idx]:
                child[idx] = p2[p2_index]

        for i in range(1, V):
            if not child[i]:
                child[i] = p2[i]

        return '0' + ''.join(child[1:]) + '0'
    
    def pmx_crossover(p1, p2):
        start, end = sorted([rand_num(1, V), rand_num(1, V)])
        child = [''] * V
        mapping = {}

        for i in range(start, end):
            child[i] = p1[i]
            mapping[p1[i]] = p2[i]

        for i in range(1, V):
            if not child[i]:
                gene = p2[i]
                while gene in mapping:
                    gene = mapping[gene]
                child[i] = gene

        for i in range(1, V):
            if not child[i]:
                child[i] = p2[i]

        return '0' + ''.join(child[1:]) + '0'
    
    # === Structura individului ===

    class Individual:
        def __init__(self):
            self.gnome = ""
            self.fitness = 0

        def __lt__(self, other):
            return self.fitness < other.fitness
        
    # === Funcții auxiliare GA ===

    def cal_fitness(gnome, mp):
        f = 0
        for i in range(len(gnome) - 1):
            d = mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48]
            if d == INT_MAX:
                return INT_MAX
            f += d
        return f
    
    def cooldown(temp):
        return (90 * temp) / 100
    
    def create_gnome():
        gnome = "0"
        while len(gnome) < V:
            temp = rand_num(1, V)
            ch = chr(temp + 48)
            if not repeat(gnome, ch):
                gnome += ch
        gnome += gnome[0]
        return gnome
    
    def generate_random_mp(n, max_cost=20, inf_prob=0.05):
        mp = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    row.append(INT_MAX if random() < inf_prob else randint(1, max_cost))
            mp.append(row)
        for i in range(n):
            for j in range(i, n):
                mp[j][i] = mp[i][j]
        return mp

    # === Algoritm TSP (GA + SA hibrid) ===
    # Această funcție implementează un algoritm evolutiv pentru rezolvarea problemei comis-voiajorului (TSP) folosind mutații și recombinări

    def TSPUtil(mp, mutation_func, crossover_func=None):
        start_time = time.time()
        gen = 1
        gen_thres = 5
        temperature = 10000

        population = []
        for _ in range(POP_SIZE):
            ind = Individual()
            ind.gnome = create_gnome()
            ind.fitness = cal_fitness(ind.gnome, mp)
            population.append(ind)

        best_individual = min(population)

        while temperature > 1000 and gen <= gen_thres:
            population.sort()
            new_population = []

            for i in range(POP_SIZE):
                p1 = population[i]

                while True:
                    if crossover_func:
                        p2 = population[rand_num(0, POP_SIZE)]
                        if p2 == p1:
                            continue
                        new_g = crossover_func(p1.gnome, p2.gnome)
                    else:
                        new_g = mutation_func(p1.gnome)

                    new_g = mutation_func(new_g)  # aplica mutația după recombinare
                    new_ind = Individual()
                    new_ind.gnome = new_g
                    new_ind.fitness = cal_fitness(new_g, mp)

                    if new_ind.fitness <= p1.fitness:
                        new_population.append(new_ind)
                        break
                    else:
                        prob = pow(2.7, -1 * ((float)(new_ind.fitness - p1.fitness) / temperature))
                        if prob > 0.5:
                            new_population.append(new_ind)
                            break

            temperature = cooldown(temperature)
            population = new_population
            current_best = min(population)
            if current_best.fitness < best_individual.fitness:
                best_individual = current_best
            gen += 1

        end_time = time.time()
        return best_individual.fitness, end_time - start_time
    

    # Crearea graficelor adaptate

    def ImplementareGraficDiversi(x, rezult):

    # Primul grafic, ce conține timpii

        fig, ax= plt.subplots()
        fig.set_size_inches(7, 3.8)
        ax.plot(x, rezult["Swap + OX"]["times"], marker = 'o', scalex = 1, scaley = 1, color = "Blue", linestyle='-')
        ax.plot(x, rezult["Inversion + PMX"]["times"], marker = 'o', scalex = 1, scaley = 1, color = "Orange", linestyle='-')
        ax.plot(x, rezult["Scramble only"]["times"], marker = 'o', scalex = 1, scaley = 1, color = "Green", linestyle='-')
        plt.plot(marker = 11, side='right')
        plt.xlabel('N')
        plt.ylabel('Timp (s)')
        plt.title('Problema comis-voiajorului, diverse tehnici, timpii rezolvărilor')
        plt.legend(["Swap + OX", "Inversion + PMX", "Scramble only"])

        canvas = FigureCanvasTkAgg(fig,master=root_tk)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0.455, rely=0.03)

        # Al doilea grafic, ce conține costurile

        fig, ax= plt.subplots()
        fig.set_size_inches(7, 3.8)
        ax.plot(x, rezult["Swap + OX"]["costs"], marker = 'o', scalex = 1, scaley = 1, color = "Blue", linestyle='-')
        ax.plot(x, rezult["Inversion + PMX"]["costs"], marker = 'o', scalex = 1, scaley = 1, color = "Orange", linestyle='-')
        ax.plot(x, rezult["Scramble only"]["costs"], marker = 'o', scalex = 1, scaley = 1, color = "Green", linestyle='-')
        plt.plot(marker = 11, side='right')
        plt.xlabel('N')
        plt.ylabel('Cost')
        plt.title('Problema comis-voiajorului, diverse tehnici, costurile rezolvărilor\n (Valoarea mai mica e mai buna)')
        plt.legend(["Swap + OX", "Inversion + PMX", "Scramble only"])

        canvas = FigureCanvasTkAgg(fig,master=root_tk)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0.455, rely=0.52)


    # === Testare ===

    def test_all_methods():
        # Definirea metodelor de mutație și recombinare
        # Swap + OX, Inversion + PMX, Scramble only
        Ns = [5, 6, 7, 8]
        methods = {
            "Swap + OX": (swap_mutation, order_crossover),
            "Inversion + PMX": (inversion_mutation, pmx_crossover),
            "Scramble only": (scramble_mutation, None)
        }

        results = {name: {"costs": [], "times": []} for name in methods}

        infDiv = ""
        matrice_fisierDiv = ""

        for N in Ns:
            global V, GENES
            V = N
            GENES = ''.join([chr(ord('A') + i) for i in range(V)])
            mp = generate_random_mp(V)

            matrice_fisierDiv += f"N = {N}\n"
            for row in mp:
                matrice_fisierDiv += str(row) + "\n"
            matrice_fisierDiv += "\n"

            for name, (mut_f, rec_f) in methods.items():
                infDiv += f"Testare {name} cu N = {N}\n"
                cost, duration = TSPUtil(mp, mut_f, rec_f)
                results[name]["costs"].append(cost)
                results[name]["times"].append(duration)

        return results, Ns, matrice_fisierDiv, infDiv

    # Preluarea valorilor din input 
    rezultate, nDiversi, matriceDiversi, infDiv = test_all_methods() 
    
    # results e dicționar de dicționare

    #results = {
    #"Swap + OX": {
        #"costs": [cost_N5, cost_N6, cost_N7, cost_N8],
        #"times": [time_N5, time_N6, time_N7, time_N8]
    #},
    #"Inversion + PMX": {
        #"costs": [...],
        #"times": [...]
    #},
    #"Scramble only": {
        #"costs": [...],
        #"times": [...]
    #}
#}


    # Apelare grafice, Ns, timpi și costuri
    ImplementareGraficDiversi(nDiversi, rezultate)

    # Afișare în fișier
    informatiiDiversi = "--- Rezultate ---"

    #informatiiDiversi += infDiv
    for method, data in rezultate.items():
        informatiiDiversi += f"\nMetoda: {method}\n"
        for metric in ["costs", "times"]:
            informatiiDiversi += f"{metric.capitalize()}: {data[metric]}\n"

    indiceFisierDiversi = indiceFisierDiversi + 1

    strFinal = f"{indiceFisierDiversi}.\n{matriceDiversi}\n{informatiiDiversi}\n------------------------------"

    # Scriere în fișier CSV
    with open("rezultat_diverse_tehnici_tsp.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([strFinal])
    

    # Afișare pop-up corespunzător
    popup = ctk.CTkToplevel(root_tk)
    popup.title("Validare N")

    popup_width = 300
    popup_height = 75

    popup.geometry(f"{popup_width}x{popup_height}")

    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()
    x = (screen_width // 2) - (popup_width // 2)
    y = (screen_height // 2) - (popup_height // 2)

    popup.geometry(f"+{x}+{y}")
    
    popup.attributes("-topmost", True)

    # Afișare mesaj de succes
    mesaj = ctk.CTkLabel(popup, text = "Diversele tehnici au fost rulate cu succes!\n"
                            "Rezultate stocate în fișier...")
    mesaj.pack(pady = 20, padx = 20)

    # Definire funcție pentru închidere
    def inchidere():
        popup.destroy()

    # Buton OK pentru închidere
    buton_ok = ctk.CTkButton(popup, text = "OK", command = inchidere)
    buton_ok.pack(pady = 10)

    # Afișare mesaj valid în label
    text_nou = "Diversele tehnici au fost rulate cu succes!\nRezultate stocate în fișier..."

    # Actualizare istoric
    global Istoric
    global NumarComanda

    Istoric = f"{Istoric}\n{str(NumarComanda)}. {text_nou}"

    label.configure(text = Istoric)

    NumarComanda += 1


# =========================
# Pop-up pentru rularea celor 4 algoritmi TSP: Backtracking, Nearest Neighbor, Simulated Annealing și Genetic Algorithm
# =========================


def popUpButon5(input4, label):
    global indiceFisier

    # Preluarea conținutului
    continut = input4.get()
    input4.delete(0, tk.END)

    # Verificarea dacă conținutul este un număr întreg
    if continut.isdigit():

    # Implementarea rezolvării cu algoritm evoluționist

        # 1. Generarea aleatorie a coordonatelor pentru cele N orașe
        #cities = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(int(continut))] # continut = N

        # Python3 implementation of the above approach
        from random import randint

        INT_MAX = 2147483647
        # Number of cities in TSP
        V = 5

        # Names of the cities
        GENES = "ABCDE"

        # Starting Node Value
        START = 0

        # Initial population size for the algorithm
        POP_SIZE = 10

        # Structure of a GNOME
        # defines the path traversed
        # by the salesman while the fitness value
        # of the path is stored in an integer


        class individual:
            def __init__(self) -> None:
                self.gnome = ""
                self.fitness = 0

            def __lt__(self, other):
                return self.fitness < other.fitness

            def __gt__(self, other):
                return self.fitness > other.fitness


        # Function to return a random number
        # from start and end
        def rand_num(start, end):
            return randint(start, end-1)


        # Function to check if the character
        # has already occurred in the string
        def repeat(s, ch):
            for i in range(len(s)):
                if s[i] == ch:
                    return True

            return False


        # Function to return a mutated GNOME
        # Mutated GNOME is a string
        # with a random interchange
        # of two genes to create variation in species
        def mutatedGene(gnome):
            gnome = list(gnome)
            while True:
                r = rand_num(1, V)
                r1 = rand_num(1, V)
                if r1 != r:
                    temp = gnome[r]
                    gnome[r] = gnome[r1]
                    gnome[r1] = temp
                    break
            return ''.join(gnome)


        # Function to return a valid GNOME string
        # required to create the population
        def create_gnome():
            gnome = "0"
            while True:
                if len(gnome) == V:
                    gnome += gnome[0]
                    break

                temp = rand_num(1, V)
                if not repeat(gnome, chr(temp + 48)):
                    gnome += chr(temp + 48)

            return gnome


        # Function to return the fitness value of a gnome.
        # The fitness value is the path length
        # of the path represented by the GNOME.
        def cal_fitness(gnome, mp):
            f = 0
            for i in range(len(gnome) - 1):
                if mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48] == INT_MAX:
                    return INT_MAX
                f += mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48]

            return f


        # Function to return the updated value
        # of the cooling element.
        def cooldown(temp):
            return (90 * temp) / 100


        # Comparator for GNOME struct.
        # def lessthan(individual t1,
        #               individual t2)
        # :
        #     return t1.fitness < t2.fitness


        # Utility function for TSP problem.
        def TSPUtil(mp):
            #Informatii pentru fisier
            informatiiGA = '------------------------------\n\nInformatii algoritm evolutionist:\n'
            # Generation Number
            gen = 1
            # Number of Gene Iterations
            gen_thres = 5

            population = []
            temp = individual()

            # Populating the GNOME pool.
            for i in range(POP_SIZE):
                temp.gnome = create_gnome()
                temp.fitness = cal_fitness(temp.gnome, mp)
                population.append(temp)

            informatiiGA += "\nInitial population: \nGNOME     FITNESS VALUE\n"

            for i in range(POP_SIZE):
                informatiiGA += f"{population[i].gnome} {population[i].fitness}\n"

            found = False
            temperature = 10000

            # Iteration to perform
            # population crossing and gene mutation.
            best_individual = min(population, key=lambda x: x.fitness)

            while temperature > 1000 and gen <= gen_thres:
                population.sort()
                informatiiGA += f"\nCurrent temp: {temperature}"
                new_population = []

                for i in range(POP_SIZE):
                    p1 = population[i]

                    while True:
                        new_g = mutatedGene(p1.gnome)
                        new_gnome = individual()
                        new_gnome.gnome = new_g
                        new_gnome.fitness = cal_fitness(new_gnome.gnome, mp)

                        #print(f"Mutație: {p1.gnome} -> {new_gnome.gnome}  |  Cost: {new_gnome.fitness}")

                        if new_gnome.fitness <= population[i].fitness:
                            new_population.append(new_gnome)
                            break
                        else:
                            prob = pow(
                                2.7,
                                -1 * ((float)(new_gnome.fitness - population[i].fitness) / temperature),
                            )
                            if prob > 0.5:
                                new_population.append(new_gnome)
                                break

                temperature = cooldown(temperature)
                population = new_population

                current_best = min(population, key=lambda x: x.fitness)
                if current_best.fitness < best_individual.fitness:
                    best_individual = current_best

                informatiiGA += f"\nGeneration: {gen}\n"
                informatiiGA += "GNOME     FITNESS VALUE\n"
                for i in range(POP_SIZE):
                    informatiiGA += f"{population[i].gnome} {population[i].fitness}\n"
                gen += 1

            # Returnare informații despre soluția găsită
            return best_individual.gnome, best_individual.fitness, informatiiGA

        # Generare matrice random, dar care să respecte condițiile
        def generate_random_mp(n, max_cost=20, inf_prob=0.05):
            from random import randint, random
            mp = []
            for i in range(n):
                row = []
                for j in range(n):
                    if i == j:
                        row.append(0)
                    else:
                        # Cu probabilitate inf_prob punem INT_MAX (nu există drum)
                        if random() < inf_prob:
                            row.append(INT_MAX)
                        else:
                            row.append(randint(1, max_cost))
                mp.append(row)

            # Aici se face matricea adiacentă
            for i in range(n):
                for j in range(i, n):
                    mp[j][i] = mp[i][j]
            
            return mp

        # Numărul de orașe, conținut = N
        V = int(continut)  
        GENES = ''.join([chr(ord('A') + i) for i in range(V)])  # "ABCDE"

       
        # Generarea matricei de distanțe între orașe, folosită în toți cei 4 algoritmi


        mp = generate_random_mp(V)

        #print("Matricea generată aleatoriu:")
        #for row in mp:
            #print(row)

        import time


        #Rularea algoritmului
        start = time.time()

        #rezultat = nQueen(int(continut))
        drumGA, costGA, informatiiGA = TSPUtil(mp)

        end = time.time()

        durataGA = end - start

        # 6. Afișare rezultate în consolă
        
        #print(f"\nDurata: {format(durata, ".8f")} s")

        # Inserarea rezultatelor în listă

        N.append(int(continut))
        TimpiGA.append(durataGA)
        CosturiGA.append(costGA)


        informatiiGA += f"\nDurata: {format(durataGA, ".8f")} s"
        informatiiGA += f"\nCostul: {drumGA}\n"
        informatiiGA += f"Solutia: {costGA}\n\n------------------------------\n"


    # 1. Implementarea rezolvării cu BACKTRACKING

        INT_MAX = 2147483647

        def totalCost(cost, visited, currPos, n, count, costSoFar, ans, path, best_path):
            # Dacă am vizitat toate orașele și putem reveni la punctul de start
            if count == n and cost[currPos][0] != 0:
                total_cost = costSoFar + cost[currPos][0]
                if total_cost < ans[0]:
                    ans[0] = total_cost
                    best_path[:] = path[:] + [0]  # Adaugă orașul de start la final
                return

            # Încercăm să vizităm fiecare oraș nevizitat
            for i in range(n):
                if not visited[i] and cost[currPos][i] != 0:
                    visited[i] = True
                    path.append(i)
                    totalCost(cost, visited, i, n, count + 1,
                            costSoFar + cost[currPos][i], ans, path, best_path)
                    path.pop()
                    visited[i] = False


        def tsp(cost):
            n = len(cost)
            visited = [False] * n
            visited[0] = True

            ans = [float('inf')]
            path = [0]  # Start de la orașul 0
            best_path = []

            totalCost(cost, visited, 0, n, 1, 0, ans, path, best_path)

            return ans[0], best_path


        mat = mp # folosim matricea generata la prima metoda

        # Rularea algoritmului
        startPRP = time.time()

        min_costBCKT, path = tsp(mat)

        endPRP = time.time()

        durataBCKT = endPRP - startPRP


        #print(f"Best path found:", ''.join(map(str, path)))#" -> ".join(map(str, path)))
        #print(f"Total cost: {min_costBCKT}")

        #print(f"\nDurata: {format(durataBCKT, ".8f")} s")

        informatiiBCKT = "------------------------------\n\nInformatii algoritm backtracking:\n"
        informatiiBCKT += f"\nDurata: {format(durataBCKT, ".8f")} s"
        informatiiBCKT += f"\nCostul: {min_costBCKT}"
        informatiiBCKT += f"\nSolutia: {''.join(map(str, path))}\n"

        # Inserarea rezultatelor în listă
        TimpiBCKT.append(durataBCKT)
        CosturiBCKT.append(min_costBCKT)
        # print(informatiiBCKT)
 

    # 2. Implementarea rezolvării cu Nearest Neighbor

        def tsp_nearest_neighbor(distante, start=0):
            n = len(distante)
            vizitat = [False] * n
            traseu = [start]
            cost_total = 0
            curent = start
            vizitat[start] = True

            for _ in range(n - 1):
                dist_minima = float('inf')
                urmatorul = -1
                for oras in range(n):
                    if not vizitat[oras] and distante[curent][oras] < dist_minima:
                        dist_minima = distante[curent][oras]
                        urmatorul = oras
                traseu.append(urmatorul)
                vizitat[urmatorul] = True
                cost_total += dist_minima
                curent = urmatorul

            # Întoarcerea la orașul de start
            cost_total += distante[curent][start]
            traseu.append(start)

            return traseu, cost_total
        
        distante = mp # folosim aceeasi matrice

        
        
        start = time.time()

        traseuNN, costNN = tsp_nearest_neighbor(distante)
        
        end = time.time()

        durataNN = end - start

        informatiiNN = ("------------------------------\n\nInformatii algoritm nearest neighbor:\n"
                        f"Durata: {format(durataNN, ".8f")} s"
                        f"\nCostul: {costNN}"
                        f"\nSolutia: {''.join(map(str, traseuNN))}\n")
        # 6. Afișare rezultate în consolă

        #print(f"Traseu urmat: {traseu}")
        #print(f"Cost total: {cost}")

        # Inserarea rezultatelor in lista

        #N.append(int(continut))
        TimpiNN.append(durataNN)
        CosturiNN.append(costNN)


    # 3. Implementarea rezolvarii folosind Calirea Simulata
   

        '''
        from simanneal import Annealer

        cities = mp#[(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(int(continut))]


        def distance(a, b):
            return math.hypot(b[0] - a[0], b[1] - a[1])
        
        class TSPAnnealer(Annealer):
            def __init__(self, state):
                super(TSPAnnealer, self).__init__(state)

            def move(self):
                a = random.randint(0, len(self.state) - 1)
                b = random.randint(0, len(self.state) - 1)
                self.state[a], self.state[b] = self.state[b], self.state[a]

            def energy(self):
                e = 0
                for i in range(len(self.state)):
                    city1 = cities[self.state[i]]
                    city2 = cities[self.state[(i + 1) % len(self.state)]]
                    e += distance(city1, city2)
                return e

        initial_state = list(range(int(continut)))
        random.shuffle(initial_state)

        tsp = TSPAnnealer(initial_state)
        tsp.steps = 10000
        tsp.Tmax = 25000.0
        tsp.Tmin = 2.5
        '''
        import random
        import math
        import copy
        import time

        class SimulatedAnnealingTSP:
            def __init__(self, coords, initial_temp=1000, cooling_rate=0.995):
                self.coords = coords
                self.N = len(coords)
                self.temperature = initial_temp
                self.cooling_rate = cooling_rate
                self.current_solution = list(range(self.N))
                random.shuffle(self.current_solution)
                self.best_solution = self.current_solution[:]
                self.best_cost = self.route_cost(self.best_solution)

            def route_cost(self, route):
                return sum(self.distance(route[i], route[(i + 1) % self.N]) for i in range(self.N))

            def distance(self, i, j):
                xi, yi = self.coords[i]
                xj, yj = self.coords[j]
                return math.hypot(xi - xj, yi - yj)

            def get_neighbor(self, route):
                a, b = random.sample(range(self.N), 2)
                neighbor = route[:]
                neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
                return neighbor

            def solve(self, max_iterations=10000):
                start_time = time.time()
                accept_count = 0
                improve_count = 0
                energy = self.best_cost

                #print(f"{'Temperature':<15}{'Energy':>10}    {'Accept':>6}   {'Improve':>8}     {'Elapsed':>8}   {'Remaining'}")

                for iteration in range(1, max_iterations + 1):
                    candidate = self.get_neighbor(self.current_solution)
                    candidate_cost = self.route_cost(candidate)

                    if candidate_cost < self.best_cost:
                        self.best_solution = candidate[:]
                        self.best_cost = candidate_cost
                        improve_count += 1

                    if self.accept(candidate_cost):
                        self.current_solution = candidate[:]
                        accept_count += 1

                    self.temperature *= self.cooling_rate

                    if iteration % (max_iterations // 1) == 0 or iteration == max_iterations:
                        elapsed = time.time() - start_time
                        remaining = elapsed / iteration * (max_iterations - iteration)
                        #print(f"{self.temperature:<15.2f}{self.best_cost:>10.2f}    {accept_count/iteration:>6.2%}   {improve_count/iteration:>8.2%}     {format_time(elapsed):>8}   {format_time(remaining)}")

                return self.best_solution, self.best_cost

            def accept(self, candidate_cost):
                if candidate_cost < self.best_cost:
                    return True
                else:
                    return random.random() < math.exp((self.best_cost - candidate_cost) / self.temperature)
        
        def format_time(seconds):
            return time.strftime('%H:%M:%S', time.gmtime(seconds))
        
        cities = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(int(continut))]

        #for i, coord in enumerate(cities):
            #print(f"  Oraș {i}: {coord}")
        #print()
        tsp = SimulatedAnnealingTSP(cities)#coords)

        # Rularea algoritmului
        start = time.time()

        #rezultat = nQueen(int(continut))
        #best_stateSA, best_energySA = tsp.anneal()
        solutionSA, costSA = tsp.solve(max_iterations=10000)

        end = time.time()

        durataSA = end - start


        informatiiSA = ("------------------------------\n\nInformatii algoritm calire simulata:\n\n"
                        f"Durata: {format(durataSA, ".8f")}\n"
                        f"Costul: {costSA}\n"
                        f"Solutia: {solutionSA}\n")
        # 6. Afișare rezultate în consolă
        
        #print("\nCea mai bună ordine de vizitare:")
        #print(best_stateSA)

        #print(f"\nLungimea totală a traseului: {round(best_energySA, 2)}")
        #print(f"\nDurata: {format(durataSA, ".8f")} s")

        #Inserarea rezultatelor in lista
        #N.append(int(continut))
        TimpiSA.append(durataSA)
        CosturiSA.append(costSA)
 

    # Scrierea rezultatelor în fișier


        matrice_fisier = ""
        for row in mp:
                matrice_fisier += str(row) + "\n"

        indiceFisier = indiceFisier + 1

        # 7. Salvarea TUTUROR rezultatelor într-un fișier CSV
        with open("rezultat_tsp.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([f"{indiceFisier}.\nN = {int(continut)}\nMatricea generata aleatoriu:\n{matrice_fisier}\n{informatiiBCKT}\n{informatiiNN}\n{informatiiSA}\n{informatiiGA}"])

        # 8. Inserarea rezultatelor într-un fișier clasic
        #f = open("rezultate.txt", "a")
        #f.write(str(continut) + " " + str(durata) + "\n")
        #f.close()

        # Pop-up confirmare funcționare cu succes
        popup = ctk.CTkToplevel(root_tk)
        popup.title("Validare N")

        popup_width = 300
        popup_height = 100#75
        popup.geometry(f"{popup_width}x{popup_height}")

        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()

        x = (screen_width // 2) - (popup_width // 2)
        y = (screen_height // 2) - (popup_height // 2)

        popup.geometry(f"+{x}+{y}")
    
        popup.attributes("-topmost", True)

        mesaj = ctk.CTkLabel(popup, text = f"Algoritmii au fost rulați cu succes! N = {continut}.\n"
                             f"Rezolvarea cu BCKT: traseu = {''.join(map(str, path))}, lungimea traseului = {round(min_costBCKT, 2)}, timp = {format(durataBCKT, '.8f')} s\n"
                             f"Rezolvarea cu NN: traseu = {''.join(map(str, traseuNN))}, lungimea traseului = {round(costNN, 2)}, timp = {format(durataNN, '.8f')} s\n"
                             f"Rezolvarea cu SA: traseu = {''.join(map(str, solutionSA))}{solutionSA[0]}, lungimea traseului = {round(costSA, 2)}, timp = {format(durataSA, '.8f')} s\n"
                             f"Rezolvarea cu GA: traseu = {drumGA}, lungimea traseului = {round(costGA, 2)}, timp = {format(durataGA, '.8f')} s\n"
                             "Rezultate stocate în fișiere...")
        mesaj.pack(pady = 20, padx = 20)

        # Definire funcție pentru închidere
        def inchidere():
            popup.destroy()

        # Buton OK pentru închidere
        buton_ok = ctk.CTkButton(popup, text = "OK", command = inchidere)
        buton_ok.pack(pady = 10)

        # Afișare mesaj valid în label
        text_nou = (f"A fost introdus un număr valid N ({continut}).\n"
                f"Rezolvarea cu BCKT: traseu = {''.join(map(str, path))}, lungimea traseului = {round(min_costBCKT, 2)}, timp = {format(durataBCKT, '.8f')} s\n"
                f"Rezolvarea cu NN: traseu = {''.join(map(str, traseuNN))}, lungimea traseului = {round(costNN, 2)}, timp = {format(durataNN, '.8f')} s\n"
                f"Rezolvarea cu SA: traseu = {''.join(map(str, solutionSA))}{solutionSA[0]}, lungimea traseului = {round(costSA, 2)}, timp = {format(durataSA, '.8f')} s\n"
                f"Rezolvarea cu GA: traseu = {drumGA}, lungimea traseului = {round(costGA, 2)}, timp = {format(durataGA, '.8f')} s\n"
                "Rezultate stocate în fișiere...")

        # Actualizare istoric
        global Istoric
        global NumarComanda

        Istoric = f"{Istoric}\n{str(NumarComanda)}. {text_nou}"

        label.configure(text = Istoric)

        NumarComanda += 1


    else:
        # Afișare mesaj și în label
        text_nou = f"A fost introdus un număr invalid N ({continut}).\nIntrodu un număr corect."

        Istoric = f"{Istoric}\n{str(NumarComanda)}. {text_nou}"

        label.configure(text = Istoric)

        NumarComanda += 1

        # Afișare pop-up cu atenționare
        popup = ctk.CTkToplevel(root_tk)
        popup.title("Validare N")

        popup_width = 300
        popup_height = 75
        popup.geometry(f"{popup_width}x{popup_height}")

        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()

        x = (screen_width // 2) - (popup_width // 2)
        y = (screen_height // 2) - (popup_height // 2)

        popup.geometry(f"+{x}+{y}")
    
        popup.attributes("-topmost", True)


        mesaj = ctk.CTkLabel(popup, text = "Introdu un număr pozitiv pentru N.")
        mesaj.pack(pady = 20, padx = 20)

        # Definire funcție pentru închidere
        def inchidere():
            popup.destroy()

        # Buton OK pentru închidere
        buton_ok = ctk.CTkButton(popup, text = "OK", command = inchidere)
        buton_ok.pack(pady = 10)


# =========================
# Implementarea interfeței grafice cu CustomTkinter
# =========================


# Implementarea interfeței
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

ctk.set_widget_scaling(1)
ctk.set_window_scaling(2)

root_tk = ctk.CTk()
#root_tk.geometry("588x350")
#root_tk.geometry("588x350+200+80")
root_tk.geometry("670x410+300+80")
root_tk.title("Aplicație")


label3 = ctk.CTkLabel(root_tk, text="Informații...", width = 518, height= 400, justify = "left", anchor = "nw", corner_radius = 10, fg_color = "#808080")
label3.place(relx = 0.02, rely = 0.20)

button1 = ctk.CTkButton(root_tk, command = lambda: popUpButon1(label3), fg_color="#4467C4", text="Afișare grafice", text_color="white", border_color="#808080", border_width=4).place(relx=0.02, rely=0.08)
#button1.pack(pady=10, side = "left")

button2 = ctk.CTkButton(root_tk, command = lambda: popUpButon2(label3), fg_color="#4467C4", text="Închidere program", text_color="white", border_color="#808080", border_width=4).place(relx=0.17, rely=0.08)

buttonIstoric = ctk.CTkButton(root_tk, command = lambda: popUpButonIstoric(label3), fg_color="#4467C4", text="Ștergere istoric", text_color="white", border_color="#808080", border_width=4).place(relx=0.32, rely=0.08)

input4 = ctk.CTkEntry(root_tk, placeholder_text="Introdu numărul N:", width = 150)
input4.place(relx=0.02, rely=0.86)

button5 = ctk.CTkButton(root_tk, command = lambda: popUpButon5(input4, label3), fg_color="#4467C4", text="Pornire algoritmi", text_color="white", border_color="#808080", border_width=4).place(relx=0.17, rely=0.86)

button6 = ctk.CTkButton(root_tk, command = lambda: popUpButon6(input4, label3), fg_color="#4467C4", text="Diverse tehnici GA", text_color="white", border_color="#808080", border_width=4).place(relx=0.32, rely=0.86)

ImplementareGrafic(N, TimpiBCKT, TimpiNN, TimpiSA, TimpiGA, CosturiBCKT, CosturiNN, CosturiSA, CosturiGA)


# =========================
# Rularea interfeței principale
# =========================


root_tk.mainloop()