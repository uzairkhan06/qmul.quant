# qmul.quant


LOAD LIBRARIES

import pulp as lp
import pandas as pd
import numpy as np
import math
import csv
import timeit

Auxiliary functions

def getRouteByEN(Route, exportNode):
    ls = []
    for r in Route:
        if r[0] == exportNode:
            ls.append((r[0], r[1], Route[(r[0], r[1])]))
    return ls

def getRouteByIN(Route, importNode):
    ls = []
    for r in Route:
        if r[0] == importNode:
            ls.append(r)
    return ls

def getVarByName(vars, name):
    ls = []
    for v in vars:
        if str(v[0]) == name:
            return v 
    print("variable " + name + " not found")
    return -1

def maxQf(v):
    if math.isnan(v):
        return float('inf')
    else:
        return v

Get initial time

start = timeit.default_timer()

Reading data # new datasets were created with the help of the datasets shared by PetroIneos

#fileName = 'step1_small_data'
#fileName = 'step1_large_data'
#fileName = 'step2_small_data'
#fileName = 'step2_large_data'
#fileName = 'step3_small_data'
#fileName = 'step3_large_data'
extension = '.xls'
file = fileName+extension

Get export node data

csv = pd.read_excel(file, sheet_name="ExportNode")

exportNodes = []

for c in csv['Export_node'].to_numpy():
    exportNodes.append(c)

exportNodes = list(exportNodes)

ls = csv['Export_node_supply'].to_numpy()
ls = [np.float64(n) for n in ls] # round numbers
supply = zip(exportNodes, ls)
supply = dict(supply)

Get import node data

#csv = pd.read_csv("importNode.csv")
csv = pd.read_excel(file, sheet_name="ImportNode")

importNodes = []

for c in csv['Import_node'].to_numpy():
    importNodes.append(c)

importNodes = list(importNodes)

ls = csv['Import_node_capacity'].to_numpy()
ls = [np.float64(n) for n in ls] # round numbers
capacity = zip(importNodes, ls)
capacity = dict(capacity)

Get Minimum flow data

csv = pd.read_excel(file, sheet_name="MinimumQt")
minQ = dict()

MIN_QT = 0
if 'Route' in csv:
    MIN_QT = 1
    g = csv.groupby(['Export node', 'Route'])
    ls = []
    for i in g:
        minQ.update({(i[0][0], i[0][1]) : [np.float64(n) for n in i[1].drop(columns=['Export node', 'Route']).to_numpy()[0]]})

else:
    MIN_QT = 2
    g = csv.groupby(['Export node'])

    ls = []
    for i in g:
        minQ.update({(i[0], 1) : [np.float64(n) for n in i[1].drop(columns=['Export node']).to_numpy()[0]]})

Get Maximum flow data

csv = pd.read_excel(file, sheet_name="MaximumQt")
maxQ = dict()

MAX_QT = 0
if 'Route' in csv:
    MAX_QT = 1
    g = csv.groupby(['Export node', 'Route'])
    ls = []
    for i in g:
        # round number
        aux = []
        for n in i[1].drop(columns=['Export node', 'Route']).to_numpy()[0]:
            if math.isnan(n) == True:
                aux.append(np.float64(n))
            else:
                aux.append(np.float64(n))

        maxQ.update({(i[0][0] , i[0][1]) : [aux][0]})
else:
    MAX_QT = 2
    g = csv.groupby(['Export node'])

    ls = []
    for i in g:
        maxQ.update({(i[0], 1) : [np.float64(n) for n in i[1].drop(columns=['Export node']).to_numpy()[0]]})

Get cost data

csv = pd.read_excel(file, sheet_name="Price")
cost = dict()

COST = 0
if 'Route' in csv:
    COST = 1
    g = csv.groupby(['Export node', 'Route'])
    ls = []
    for i in g:
        cost.update({(i[0][0], i[0][1]) : [np.float64(n) for n in i[1].drop(columns=['Export node', 'Route']).to_numpy()[0]]})
else:
    COST = 2
    g = csv.groupby(['Export node'])
    ls = []
    for i in g:
        cost.update({(i[0], 1) : [np.float64(n) for n in i[1].drop(columns=['Export node']).to_numpy()[0]]})

Get bottleneck Maximum flow data

OK_BN = False
try:
    csv = pd.read_excel(file, sheet_name="BotleneckMaxFlow")
    bnGroups = dict()
    for i in csv.iterrows():
        bnGroups.update({i[1]['Bottleneck']: (i[1]['Measure'],i[1]['Qt'],[])})

    print(bnGroups)
    OK_BN = True
except:
    print('no bottleneck data')
    pass

Get bottleneck data

if OK_BN == True:
    csv = pd.read_excel(file, sheet_name="Bottlenecks")

    for i, r in enumerate(csv.iterrows()):
        row = r[1]
        for j, c in enumerate(row):
            if c in bnGroups:
                bnGroups[c][2].append((row[0], importNodes[j - 2], row[1]))

    print(bnGroups)

Building model

model = lp.LpProblem(name="model")

Setting x variable
xijk = amount of goods exported from i to j through route k

xijk = []
for i in exportNodes:
    routes = getRouteByEN(cost, i)
    for k in range(len(routes)):
        for n, j in enumerate(importNodes):        
            xijk.append((lp.LpVariable( cat='Continuous', lowBound=0, name="x(" + str(i) + "," + str(j) + "," + str(k + 1) + ")"), routes[k][2][n]))

for x in xijk:
    model.addVariable(x[0])

Setting objective function

fo = 0

for x in xijk:
    fo += x[0] * x[1]

# minimize
model.sense = lp.LpMinimize 
# setting the objective
model.setObjective(fo)

Setting constraints
Export nodes supply constraints

for en in exportNodes:
    c1 = lp.LpConstraint()
    routes = getRouteByEN(cost, en)
    for k in routes:
        for iN in importNodes:
            name = "x(" + en + "," + iN + "," + str(k[1]) + ")"
            x = getVarByName(xijk, name)
            if(c != -1):
                c1 += x[0]
    model.addConstraint(c1 == supply[en], name="c1(" + str(en) + ")")

Import nodes capacity constraints

for iN in importNodes:
    c2 = lp.LpConstraint()    
    for en in exportNodes:
        routes = getRouteByEN(cost, en)        
        for k in routes:
            name = "x(" + en + "," + iN + "," + str(k[1]) + ")"
            x = getVarByName(xijk, name)
            if(x != -1):
                c2 += x[0]
    model.addConstraint(c2 <= capacity[iN], name="c2(" + iN + ")")

Route maximum capacity constraints

for i, en in enumerate(exportNodes):
    for j, iN in enumerate(importNodes):
        routes = getRouteByEN(cost, en) 
        for r in routes:
            name = "x(" + en + "," + iN + "," + str(r[1]) + ")"
            x = getVarByName(xijk, name)
            Max = maxQ[(en, r[1])][j]
            if math.isnan(Max) == False:
                model.addConstraint(x[0] <= Max, name='c3(' + en + ',' + iN + ',' + str(r[1]) + ')')
            else:
                pass

Route minimum capacity constraints

if MIN_QT == 1:
    for i, en in enumerate(exportNodes):
        for j, iN in enumerate(importNodes):
            routes = getRouteByEN(cost, en) 
            for r in routes:
                name = "x(" + en + "," + iN + "," + str(r[1]) + ")"
                x = getVarByName(xijk, name)
                Min = minQ[(en, r[1])][j]
             
                if math.isnan(Min) == False:
                    model.addConstraint(x[0] >= Min, name='c4(' + en + ',' + iN + ',' + str(r[1]) + ')')
                else:
                    pass

if MIN_QT == 2:
    for i, en in enumerate(exportNodes):
        for j, iN in enumerate(importNodes):
            routes = getRouteByEN(cost, en) 
            c = lp.LpConstraint()
            for r in routes:
                name = "x(" + en + "," + iN + "," + str(r[1]) + ")"
                x = getVarByName(xijk, name)[0]          
                c += x

            Min = minQ[(en, 1)][j] 
            if math.isnan(Min) == False:
                model.addConstraint(c >= Min, name='c4(' + en + ',' + iN + ')')
            else:
                pass
                

Bottleneck constraints

if OK_BN == True:
    for i in bnGroups:
        maxFlow = bnGroups[i][1]
        c = 0
        for j in bnGroups[i][2]:
            name = 'x(' + j[0] + ',' + j[1] + ',' + str(j[2]) + ')'
            x = getVarByName(xijk, name)[0]
            c += x

        model.addConstraint(c <= maxFlow, name='c5(' + i + ')')

#model

modelFile = fileName+'-Model.lp'
model.writeLP(modelFile)
print()

Solve the model

try:
    model.solve()
    print('solved')
except:
    print('solver crashed')

print("Status: ", lp.LpStatus[model.status], "\n")

FO = model.objective.value()
print("Value: ", FO, "\n")

Get runtime

finish = timeit.default_timer()
print(finish - start, ' seconds')

Sensitivity analisys

d = dict()
for i in exportNodes:
    s = 0
    routes = getRouteByEN(minQ, i)
    
    for k in routes:
        for j in k[2]:
            if math.isnan(j):
                s += 0
            else:
                s += float(j)
    d.update({i: s})

d = dict()
for i in exportNodes:
    s = 0
    routes = getRouteByEN(maxQ, i)
    for k in routes:
        for j in k[2]:
            if math.isnan(j):
                s += 100.0
            else:
                s += float(j)
    d.update({i: s})

d = dict()
for j, iN in enumerate(importNodes):
    s = 0
    for en in exportNodes:
        routes = getRouteByEN(minQ, en)
        for r in routes:
            #s += r[2][j]
            if math.isnan(r[2][j]):
                s += 100.0
            else:
                s += r[2][j]
    d.update({iN: s}) 

d = dict()
for j, iN in enumerate(importNodes):
    s = 0
    for en in exportNodes:
        routes = getRouteByEN(maxQ, en)
        for r in routes:
            #s += r[2][j]
            if math.isnan(r[2][j]):
                s += 100.0
            else:
                s += r[2][j]
    d.update({iN: s}) 

sensitivAnalysisFile = fileName+'-SA.xlsx'
writer = pd.ExcelWriter(sensitivAnalysisFile)

ls = []
for name, c in model.constraints.items():
    r = [name, c.pi, c.slack]
    ls.append(r)

sp = pd.DataFrame(ls, columns=['constrain', 'shadow price', 'slack'])
sp.to_excel(writer, sheet_name='constraints', index=False)

ls = []
for x, value in xijk:
    r = [x.name, x.varValue, x.dj]
    ls.append(r)

sp = pd.DataFrame(ls, columns=['variable', 'value', 'reduced cost'])
sp.to_excel(writer, sheet_name='variables', index=False)

writer.save()

Save output data as a table

data = []
for iN in importNodes:
    ls = []
    for en in exportNodes:        
        routes = getRouteByEN(cost, en)
        for k in routes:	        
            name = "x(" + en + "," + iN + "," + str(k[1]) + ")"
            x = getVarByName(xijk, name)[0]
            ls.append(x.value())
    data.append(ls)

data = []
for en in exportNodes:    
    routes = getRouteByEN(cost, en)
    for k in routes:
        ls = []
        ls.append(k[0])
        ls.append(k[1])
        for iN in importNodes:
            name = "x(" + en + "," + iN + "," + str(k[1]) + ")"
            x = getVarByName(xijk, name)[0]
            ls.append(x.value())
        data.append(ls)

data[0].append(FO)
for d in data[1:]:
    d.append("")

output = pd.DataFrame(data, columns=['Export node', 'Route'] + importNodes + ['Total cost'])
#output

resultFile = fileName+'-Result.xlsx'
try:
    output.to_excel(resultFile, index=False)
except:   
    print("Error writing file")

Get runtime

finish_final_code = timeit.default_timer()

print(finish_final_code - start, ' seconds')

