import numpy as np
import matplotlib.pyplot as plt
# for num_units = 200
# p_drop_rate = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4]
# test_loglik_1 = [1.528,  1.520, 1.530, 1.514, 1.526, 1.521, 1.521, 1.524, 1.5206, 1.499, 1.476, 1.448]

# for p_drop_rate = 0.04
# num_units = [10,20,30,50,100,150,200,250]
# test_loglik_2 = [1.340,1.426,1.467,1.495,1.525,1.536,1.537,1.525]

# test_rmse_1 = [0.057,0.058,0.057,0.058,0.057,0.057,0.057,0.057,0.058,0.057,0.058,0.060]
# test_rmse_2 = [0.068,0.063,0.06,0.059,0.057,0.057,0.057,0.057]


# # for num_units = 10
# p_drop_rate = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]
# test_loglik_1 = [1.28405, 1.27515, 1.24707, 1.19988, 1.13111, 1.04360, 1.00609]
# test_rmse_1 = [0.07320,0.07309, 0.07401, 0.07610, 0.08056, 0.08770, 0.08965]

# for num_units = 50
p_drop_rate = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]
test_loglik_1 = [1.45164, 1.46127, 1.45386, 1.44570, 1.41555, 1.38638, 1.33715]
test_rmse_1 = [0.06167,0.05979,0.06006,0.06010,0.06160,0.06239,0.06546]

# for p_drop_rate = 0.1
num_units = [10,20,30,50,100]
test_loglik_2 = [1.3385,1.39516,1.43582,1.47033,1.49213]
test_rmse_2 = [0.07045, 0.06545, 0.06293, 0.06125, 0.05956]


plt.figure()
plt.plot(p_drop_rate, test_loglik_1, 'b', label='Observations');
plt.xlabel('Dropout rate', fontsize=16)
plt.ylabel('Test loss', fontsize=16)
plt.show()

plt.figure()
plt.plot(num_units, test_loglik_2, 'b', label='Observations');
plt.xlabel('Number of neurons', fontsize=16)
plt.ylabel('Test loss', fontsize=16)
plt.show()

plt.figure()
plt.plot(p_drop_rate, test_rmse_1, 'b', label='Observations');
plt.xlabel('Dropout rate', fontsize=16)
plt.ylabel('Test RMSE', fontsize=16)
plt.show()

plt.figure()
plt.plot(num_units, test_rmse_2, 'b', label='Observations');
plt.xlabel('Number of neurons', fontsize=16)
plt.ylabel('Test RMSE', fontsize=16)
plt.show()
