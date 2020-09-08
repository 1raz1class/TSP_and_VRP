import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools as itr
import scipy.spatial.distance as ssd
import ObjectiveValue as OB



### --> DYNAMIC TABU LIST WITH ASPIRATION BEST SO FAR and MULTI-RESTART WITH MUTATION <-- ###
### Returns back to the best-so-far after some iterations

print()
print("DYNAMIC TABU LIST WITH ASPIRATION BEST SO FAR and MULTI-RESTART WITH MUTATION ")


# X coordinates
X_Coordinations = np.array([82,82,82,82,82,82,82,82,82,82,96,50,49,13,29,58,84,14,2,3,5,
                            98,84,61,1,88,91,19,93,50,98,5,42,61,9,80,57,23,20,85,98])

# Y coordinates
Y_Coordinations = np.array([76,76,76,76,76,76,76,76,76,76,44,5,8,7,89,30,39,24,39,82,10,
                            52,25,59,65,51,2,32,3,93,14,42,9,62,97,55,69,15,70,60,5])

# Stack them on top of each other
X_Y_Coordinations = np.column_stack((X_Coordinations,Y_Coordinations))

# Create a distance matrix from the Euclidean distance
Distance_Matrix = ssd.cdist(X_Y_Coordinations,X_Y_Coordinations,'euclidean')

Capacities = np.array([0,0,0,0,0,0,0,0,0,0,19,21,6,19,7,12,16,6,16,8,14,21,16,3,22,18,
                       19,1,24,8,12,4,8,24,24,2,20,15,2,14,9])

# Create a distance dataframe
Dist_Dataframe = pd.DataFrame(Distance_Matrix,columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,
                                                       14,15,16,17,18,19,20,21,22,23,24,
                                                       25,26,27,28,29,30,31,32,33,34,35,
                                                       36,37,38,39,40,41],
                                            index=[1,2,3,4,5,6,7,8,9,10,11,12,13,
                                                       14,15,16,17,18,19,20,21,22,23,24,
                                                       25,26,27,28,29,30,31,32,33,34,35,
                                                       36,37,38,39,40,41])
# Create a capacity dataframe
Cap_Dataframe = pd.DataFrame(Capacities,
                             columns=["Capacity"],index=[1,2,3,4,5,6,7,8,9,10,11,12,13,
                                                       14,15,16,17,18,19,20,21,22,23,24,
                                                       25,26,27,28,29,30,31,32,33,34,35,
                                                       36,37,38,39,40,41])


# Initial solution to start with
# Initial solution to start with\n",
X0 = [24,26,22,31,11,16,25,30,34,12,14,1,20,39,15,5,36,13,18,19,32,
                38,28,10,9,23,33,21,7,4,40,35,2,17,8,6,37,3,29,27,41]
     
### For Documentation ###
Initial_For_Final = [24,26,22,31,11,16,25,30,34,12,14,1,20,39,15,5,36,13,18,19,32,
             38,28,10,9,23,33,21,7,4,40,35,2,17,8,6,37,3,29,27,41]


### VARIABLES ###
Runs = 2500
P_Mutation = 0.2

### TABU LIST ###
Length_of_Tabu_List = 20

Tabu_List = np.empty((0,len(X0)+1)) # +1 to keep track of the fitness value

Final_Solution = []


# Initial random solution
rnd_sol_1 = X0[:]


Total_Distance_Initial = OB.Complete_Distance_Not_Random(rnd_sol_1) # Get the distance of the initial solution

print()
print("Initial Distance:", Total_Distance_Initial) # Print the initial distance
print("Initial Random Solution:", rnd_sol_1) # Print the initial solution

# Save the solutions so you can pick the best later
Index_For_Best_1 = np.empty((0,2))
Save_Solutions_Here = np.empty((0,len(X0)+1)) 

Index_For_Best_2 = np.empty((0,2))


Best_So_Far = []

Iterations = 1

for i in range(Runs):
    
    print()
    print("--> This is the %i" % Iterations, "th Iteration <--")
    
    
    List_of_N = list(itr.combinations(rnd_sol_1, 2)) # Get all the combinations of every two to swap
    
    Counter_for_N = 0
    t = 0
    r = 1
    
    Store_all_Combinations = np.empty((0,len(X0))) # To store all possible combinations
    
    for i in List_of_N: # For i in every combintation of 2's
        X_Swap = []
        A_Counter = List_of_N[Counter_for_N] # Each element in the combination of 2's
        A_1 = A_Counter[0] # First element of the combination
        A_2 = A_Counter[1] # Second element of the combination
        
        # Making a new list of the new set of solutions, with 2-opt (swap)
        u= 0
        for j in rnd_sol_1:
            if rnd_sol_1[u] == A_1: # If the solution's u-th element is the s element in the combination
                X_Swap = np.append(X_Swap,A_2) # Swap with second element
            elif rnd_sol_1[u] == A_2:
                X_Swap = np.append(X_Swap,A_1)
            else:
                X_Swap = np.append(X_Swap,rnd_sol_1[u]) # If not first or second, put the element as is
            
            X_Swap = X_Swap[np.newaxis] # Change axis to horizontal
    
            u = u+1
        
        
        Store_all_Combinations = np.vstack((Store_all_Combinations,X_Swap)) # Stack all the solutions, all possible combinations
        
        Counter_for_N = Counter_for_N+1
    
    
    OF_Values_for_N_i = np.empty((0,len(X0)+1))
    OF_Values_all_N = np.empty((0,len(X0)+1))
    
    N_Count = 1
    
    # Calculating OF for the i-th solution in N
    for i in Store_all_Combinations:
        
        Cost = OB.Complete_Distance_Not_Random(i) # The distance function
        
        Penalty_1 = OB.Penalty_1(i,20) # The penalty function for carrying more than 100
        Penalty_2 = OB.Penalty_2(i,200,300) # The penalty of warehouses' locations
        Penalty_3 = OB.Penalty_3(i,400) # The penalty of warehouses' locations
        
        
        Total_Distance_N_i = Cost + Penalty_1 + Penalty_2 + Penalty_3 # Add the costs to the penalties
        
        
        i = i[np.newaxis] # Change axis to horizontal
        
        OF_Values_for_N_i = np.column_stack((Total_Distance_N_i,i)) # Put the distance next to its solution
        
        OF_Values_all_N = np.vstack((OF_Values_all_N,OF_Values_for_N_i)) # Stack them on top of each other
        
        N_Count = N_Count+1
        
        
    # Ordered OF of neighborhood 
    OF_Values_all_N_Ordered = np.array(sorted(OF_Values_all_N,key=lambda x: x[0])) # Sort from lowest distance to highest
    
    
    # Get best so far
    if len(Save_Solutions_Here) >= 1: # Only if the "Save Solutions Here" had data in it
        t = 0
        Best_Here = []
        for i in Save_Solutions_Here:
            
            if (Save_Solutions_Here[t,0]) <= min(Save_Solutions_Here[:,0]): # Get the minimum distance
                Best_Here = Save_Solutions_Here[t,:]
            t = t+1
        
        Best_So_Far = Best_Here # Get best so far solution in the "Save Solutions Here"
        
    
    # Check if solution is already in Tabu list, if yes, choose the next one
    t = 0
    Solution_in_Hand = OF_Values_all_N_Ordered[t] # First element in the ordered list
    
    
    # Aspiration Criteria
    if len(Save_Solutions_Here) >= 1:
        
        if Solution_in_Hand in Tabu_List: # If the solution is already in the tabu list
            while Solution_in_Hand[0] in Tabu_List[:,0]: # The distance of the solution is the same as one in the tabu list
                if Solution_in_Hand[0] < Best_So_Far[0]: # If it is less, then...
                    Solution_in_Hand = Best_So_Far # It becomes best so far
                    break # Then break
                else:
                    Solution_in_Hand = OF_Values_all_N_Ordered[t] # If it wasn't less, the current solution becomes what was chosen in the list
                    t = t+1 # Add one to move to the next element
        else:
            Solution_in_Hand = Best_So_Far # If solution was not in tabu list, the best so far becomes the current solution
          
        if len(Tabu_List) >= Length_of_Tabu_List: # If the tabu list was full
            
            if Tabu_List[0,0] != Solution_in_Hand[0]: # If the first solution distance is not equal to the current solution you are at
                Difference = len(Tabu_List) - Length_of_Tabu_List # You want to remove one
                Diff = (Difference+1)
                for i in range(Diff):
                    Tabu_List = np.delete(Tabu_List, (Length_of_Tabu_List-1), axis=0) # Remove the last/oldest solution
                    Tabu_List = np.vstack((Solution_in_Hand,Tabu_List)) # Put the solution in hand in the tabu list
                
            else:
                Tabu_List[0,:] = Solution_in_Hand[np.newaxis] # The solution in hand becomes instead of the first element
        else:
                Tabu_List = np.vstack((Solution_in_Hand,Tabu_List)) # If the tabu list was not full, put the solution in hand in it
    
    
    Save_Solutions_Here = np.vstack((Solution_in_Hand,Save_Solutions_Here)) # Save the solution in hand in a list to keep track of it
    
    Index_For_Best_1 = np.hstack((Solution_in_Hand[0],Iterations)) # Keep track of the iteration the best solution was achieved at
    Index_For_Best_2 = np.vstack((Index_For_Best_1,Index_For_Best_2)) # Stack them on top of each other
    
    
    
    # In order to "kick-start" the search when stuck in a local optimum

    Mod_Iterations_1 = Iterations%20 # For changing the tabu list size
    Mod_Iterations_2 = Iterations%40 # For changing the best-so-far solution
    Mod_Iterations_3 = Iterations%30 # For returning to the best-so-far
    
    
    
    if Mod_Iterations_3 == 0: # To return to the best so far in case you kept getting worse solutions
        if Solution_in_Hand[0] > Best_So_Far[0]: # If where you are is worse than the best so far
            Solution_in_Hand = Best_So_Far # Become best so far
            print()
            print("Go Back to Best-So-Far")
            print()
        else:
            Solution_in_Hand = Solution_in_Hand # Else, remain where you are
    
    Random_1 = np.random.rand() # Take a random number to mutate
    
    if Random_1 <= P_Mutation: # Probablity is less than the probablity of mutation
        '''
        if Mod_Iterations == 0:
        '''   
        Mutated_Solution_in_Hand = []

        # ["2000","2","5","6","9","1"]
        
        Ran_Mut_1 = np.random.rand() # Probablity to Mutate
        Ran_Mut_2 = np.random.randint(1,len(X0)+1) # Random integer in the length of the solution
        Ran_Mut_3 = np.random.randint(1,len(X0)+1)
        
        A1 = Ran_Mut_2 # A1 is a random integer
        A2 = Ran_Mut_3 # A2 is a random integer
        
        while A1 == A2:
            A2 = np.random.randint(1,len(X0)+1) # If A1 is equal to A2, repeat
        
        
        if A1 < A2: # If A1 comes before A2
            M_Child_1_Pos_1 = Solution_in_Hand[A1] # Index A1 in the solution
            M_Child_1_Pos_2 = Solution_in_Hand[A2] # Index A2 in the solultion
            A2 = A2+1
            Rev_1 = Solution_in_Hand[:]
            Rev_2 = list(reversed(Solution_in_Hand[A1:A2])) # Reverse the order
            t = 0
            for i in range(A1,A2):
                Rev_1[i] = Rev_2[t]
                t = t+1
            
            Mutated_Solution_in_Hand = Rev_1[:]
        
        else:
            M_Child_1_Pos_1 = Solution_in_Hand[A2] # Index A2 in the solution
            M_Child_1_Pos_2 = Solution_in_Hand[A1] # Index A1 in the solution
            A1 = A1+1
            Rev_1 = Solution_in_Hand[:]
            Rev_2 = list(reversed(Solution_in_Hand[A2:A1])) # Reverse the order
            t = 0
            for i in range(A2,A1):
                Rev_1[i] = Rev_2[t]
                t = t+1
            
            Mutated_Solution_in_Hand = Rev_1[:]
        
       
        Ran_1 = np.random.randint(1,len(X0)+1) # Random integers in length of solution
        Ran_2 = np.random.randint(1,len(X0)+1)
        Ran_3 = np.random.randint(1,len(X0)+1)
        Ran_4 = np.random.randint(1,len(X0)+1)
        
        
        Xt = []
        A1 = Mutated_Solution_in_Hand[Ran_1] # Get a random integer in the mutated solution
        A2 = Mutated_Solution_in_Hand[Ran_2] # Get a random integer in the mutated solution
        
        # Making a new list of the new set of nodes
        T_Temp = Mutated_Solution_in_Hand
        
        w = 0
        for i in T_Temp:
            if T_Temp[w] == A1:
                Xt=np.append(Xt,A2)
            elif T_Temp[w] == A2:
                Xt=np.append(Xt,A1)
            else:
                Xt=np.append(Xt,T_Temp[w])
            w = w+1
        
        Mutated_Solution_in_Hand = Xt
        
        # Same nodes gets switched
        
        Xt = []
        A1 = Mutated_Solution_in_Hand[Ran_1]
        A2 = Mutated_Solution_in_Hand[Ran_3]
        
        # Making a new list of the new set of nodes
        w = 0
        for i in Mutated_Solution_in_Hand:
            if Mutated_Solution_in_Hand[w] == A1:
                Xt=np.append(Xt,A2)
            elif Mutated_Solution_in_Hand[w] == A2:
                Xt=np.append(Xt,A1)
            else:
                Xt=np.append(Xt,Mutated_Solution_in_Hand[w])
            w = w+1
        
        Solution_in_Hand = Xt
        
        
        
    rnd_sol_1 = Solution_in_Hand[1:]

    
    # Aspiration Criteria, Diversification, to Encourage the Search to Diversify
    
    if Mod_Iterations_2 == 0:
        Ran_1 = np.random.randint(1,len(X0)+1) # Get a random integer
        Ran_2 = np.random.randint(1,len(X0)+1)
        Best_So_Far[Ran_1],Best_So_Far[Ran_2] = Best_So_Far[Ran_2],Best_So_Far[Ran_1]
        Ran_1 = np.random.randint(1,len(X0)+1)
        Best_So_Far[Ran_1],Best_So_Far[Ran_2] = Best_So_Far[Ran_2],Best_So_Far[Ran_1]
        Ran_3 = np.random.randint(1,len(X0)+1)
        Ran_4 = np.random.randint(1,len(X0)+1)
        Best_So_Far[Ran_3],Best_So_Far[Ran_4] = Best_So_Far[Ran_4],Best_So_Far[Ran_3]
        Ran_3 = np.random.randint(1,len(X0)+1)
        Best_So_Far[Ran_3],Best_So_Far[Ran_4] = Best_So_Far[Ran_4],Best_So_Far[Ran_3]
        rnd_sol_1 = Best_So_Far[1:]
    
    '''
    print()
    print("New Solution_in_Hand:", Solution_in_Hand)
    '''
    Iterations = Iterations+1

    # Change length of Tabu List every 20 runs, between 5 and 25
    if Mod_Iterations_1 == 0:
        Length_of_Tabu_List = np.random.randint(10,26)
        '''
        print()
        print("Length of Tabu List New:",Length_of_Tabu_List)
        '''


t = 0
Final_Here = []
for i in Save_Solutions_Here:
    
    if (Save_Solutions_Here[t,0]) <= min(Save_Solutions_Here[:,0]):
        Final_2 = []
        Final_2 = [Save_Solutions_Here[t,0]]
        Final_Here = Save_Solutions_Here[t,:]
    t = t+1
        
A_2_Final = min(Save_Solutions_Here[:,0])

Final_Solution = Final_Here[np.newaxis]


t = 0
Index_Here = []
for i in Index_For_Best_2:
    
    if (Index_For_Best_2[t,0]) <= min(Index_For_Best_2[:,0]):
        Final_3 = []
        Final_3 = [Index_For_Best_2[t,0]]
        Index_Here = Index_For_Best_2[t,:]
    t = t+1
        
A_3_Final = min(Index_For_Best_2[:,0])

Index_for_Best = Index_Here[1]


# Show truck capacities
Temp_Cap = Final_Solution[0,1:]
To_Save_Capacities = []
t = 0
for j in Temp_Cap:
    if t < 41:
        To_Save_Capacities = np.append(To_Save_Capacities,Temp_Cap[t])
        t = t+1
'''
print()
print("Cities:",To_Save_Capacities)
'''
Truck_Cap_1_M_1,Truck_Cap_2_M_1 = [],[]    
for j in To_Save_Capacities: # Only the values between 1 and 10 (warehouses)
    if 1 <= j <= 10:
        Truck_Cap_1_M_1 = np.append(Truck_Cap_1_M_1,j)
for i in Truck_Cap_1_M_1: # The warehouses' indexes
    value_index = np.where(To_Save_Capacities == i)
    Truck_Cap_2_M_1 = np.append(Truck_Cap_2_M_1,value_index)
t = 0
Caps_All, Caps_Sum_One_By_One  = [],[]
for k in range(len(Truck_Cap_2_M_1)):
    if t < 9:
        Ind_1 = int(Truck_Cap_2_M_1[t])+1
        Ind_2 = int(Truck_Cap_2_M_1[t+1])
        if Ind_1 in range(11) and Ind_2 in range(11):
            Truck_Cap_3_M_1_1 = 0
        Truck_Cap_3_M_1_1 = To_Save_Capacities[Ind_1:Ind_2]
        u = 0
        '''
        print()
        print("Cities between Two WH:",Truck_Cap_3_M_1_1)
        '''
        Caps_One_By_One = []
        if len(Truck_Cap_3_M_1_1) != 0:
            for o in Truck_Cap_3_M_1_1:
                Caps_Sum = 0
                Capacity_1 = Truck_Cap_3_M_1_1[u]
                Capacity_2 = Cap_Dataframe.loc[Capacity_1,'Capacity']
                Caps_One_By_One = np.append(Caps_One_By_One,Capacity_2)
                Caps_All = np.append(Caps_All,Capacity_2)
                u = u+1
                Caps_Sum = sum(Caps_One_By_One)
            Caps_Sum_One_By_One = np.append(Caps_Sum_One_By_One,Caps_Sum)
    t = t+1


Without_Penalty = Final_Solution[:,1:].tolist()

Without_Penalty_Cost = OB.Complete_Distance_Not_Random(Without_Penalty[0])


print()
print()
print("DYNAMIC TABU LIST WITH ASPIRATION BEST SO FAR and MULTI-RESTART WITH RANDOM NUMBER FOR MUTATION")
print()
print("Initial Solution:",Initial_For_Final)
print("Initial Distance:", Total_Distance_Initial)
print()
print("Min in all Iterations:",Final_Solution[:,1:])
print()
print("The Lowest Distance is:",Final_Solution[:,0])
print("The Lowest Distance Without Penalties is:",Without_Penalty_Cost)
print()
print("Trucks' Capacities:",Caps_Sum_One_By_One)
print()
print("At Iteration #:",Index_for_Best)



To_Plot = np.flip(Save_Solutions_Here, 0)


Look = (Final_Solution[:,0]).tolist()
Look = float(Look[0])
Look = int(Look)

Without_Penalty_Cost = int(Without_Penalty_Cost)

plt.plot(To_Plot[:,0])
plt.axhline(y=Look,color="r",linestyle='--')
plt.title("Total Distance Traveled by Trucks",fontsize=20,fontweight='bold')
plt.xlabel("Iterations",fontsize=18,fontweight='bold')
plt.ylabel("Distance (KM)",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
xyz=(Iterations/2.6, Look)
xyzz = (Iterations/2.5, Look+200)
plt.annotate("Minimum Reached at: %s" % Without_Penalty_Cost, xy=xyz, xytext=xyzz,
             arrowprops=dict(facecolor='black', shrink=0.001,width=1,headwidth=5),
             fontsize=12,fontweight='bold')
plt.show()
