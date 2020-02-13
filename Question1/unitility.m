plot(out_lm_f2, 'b')
hold  on
plot(testYF2, 'r')
hold  on
plot(mse_lm_f2, 'g')

min(mse_lm_f2)
mean(mse_lm_f2)
max(mse_lm_f2)
std(mse_lm_f2)