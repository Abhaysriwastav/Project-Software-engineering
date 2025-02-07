import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class StockVisualizer:
    def __init__(self):
        self.plt = plt
        
    def plot_predictions(self, y_true, y_pred, title):
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        
        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # Encode
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        
        return graphic