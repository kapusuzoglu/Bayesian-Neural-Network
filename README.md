# Bayesian Neural Network
Implement Bayesian Neural Network (BNN) using Pytorch to predict mean and both aleatoric and epistemic uncertainties for the quantity of interest (QoI). In Bayesian setting, there are two main types of uncertainty; *aleatoric* uncertainty, which captures the noise inherent in the observations and *epistemic* uncertainty that accounts for the uncertainty in the model. By collecting more data we can reduce the epistemic uncertainty. However, aleatoric uncertainty cannot be reduced even if more data were to be collected.

## Pytorch_BNN_v1.py
1) Import the data
2) Build the MC dropout BNN using Pytorch
3) Use K-fold (shuffle the data as well)
4) Train the model with the training data and test using the test data
5) Predict unobserved data and plot

## Pytorch_BNN_v1_pdropCheck.py
Loop over different dropout probabilities and different number of units to find the optimal values.

## plot_performance.py
Plots RMSE and log likelihood of test data with respect to number of neurons and dropout probability rate.

## BNN_v1_pdropCheck_plot_leave_one_out
Plots file "Part25_Predictions", which shows the average bond length predictions of 3 sets of part 25.

## BNN_v1_pdropCheck
Plots: 

*Predicted bond length (mm)* vs *Observed bond length (mm)*
*Temperature* vs *Bond length (mm)*

using the test data.