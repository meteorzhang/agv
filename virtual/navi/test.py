import math

longitude = 0
latitude = 0
distance = 1
omg = 45
x=latitude+distance * math.cos(math.radians(omg))
y=longitude+distance * math.sin(math.radians(omg))
print(x,y)




