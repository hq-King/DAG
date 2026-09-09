import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('csv.csv')

first_time = data.iloc[0]['Wall time']
data['Wall time'] = data['Wall time'] - first_time

plt.plot(data['Wall time'], data['Value'], label = "loss-time")
plt.plot(data['Step'], data['Value'], label = "loss-step")

plt.title('training loss-time')
plt.xlabel('Time(s)')
plt.ylabel('Value')


plt.title('Training Loss')
plt.xlabel('Time(s) / Step')
plt.ylabel('Value')

plt.legend()
plt.grid()

plt.savefig('loss.jpg')



