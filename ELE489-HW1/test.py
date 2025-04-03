from ucimlrepo import fetch_ucirepo 

 
# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 

print("the data")
print(X)
print("the output data")
print(y)

# metadata 
print(wine.metadata) 
  
# variable information 
print(wine.variables) 
