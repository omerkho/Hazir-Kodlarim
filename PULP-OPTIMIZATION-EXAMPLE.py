# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:07:55 2020

@author: Omer
"""

import pulp

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

model = LpProblem(name= "small-problem", sense=LpMaximize)

x = LpVariable(name="x",lowBound=  0)
y= LpVariable(name="y",lowBound=   0)

model += (2*x + y <= 20, "red_cons")
model += (4*x - 5*y >= -10,"blue_cons")
model += (-x + 2*y >= -2, "yellow_cons")
model += (-x + 5*y == 15, "green_cons")

obj_func = x+ 2*y
model += obj_func

model

status = model.solve()

model.objective.value()

for var in model.variables(): 
    print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items(): 
    print(f"{name}: {constraint.value()}")

model.solver
