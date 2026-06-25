gamma = 0.9
R1 = -1
R2 = -0.5
R3 = 2.5
R4 = 1
R5 = 3

G5 = 0
G4 = R5 + G5
G3 = R4 + gamma * G4
G2 = R3 + gamma * G3
G1 = R2 + gamma * G2
G0 = R1 + gamma * G1 

print("G0 = {},   G1 = {},   G2 = {},   G3 = {},   G4 = {},   G5 = {}".format(G0,G1,G2,G3,G4,G5))