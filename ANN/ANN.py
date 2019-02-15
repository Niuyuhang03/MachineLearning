import numpy as np
import pandas as pd

class Loader():
    '''
    数据加载器基类
    '''
    def __init__(self, path, count):
        self.path = path
        self.count = count

    def get_file_content(self):
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

class ImageLoader(Loader):
    '''
    读取特征信息
    '''
    def load(self):
        content = self.get_file_content()
        data = []
        for index in range(self.count):
            start = index * 28 * 28 + 16
            picture = []
            for i in range(28):
                for j in range(28):
                    picture.append(content[start + i * 28 + j])
            data.append(picture)
        return data

class LabelLoader(Loader):
    '''读取标签信息'''
    def load(self):
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            label_vec = []
            label = index + 8
            for i in range(10):
                if i == label:
                    label_vec.append(0.9)
                else:
                    label_vec.append(0.1)
            labels.append(label_vec)
        return labels

class FullConnectedLayer():
    '''
    全连接层实现类
    '''
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        self.input = np.array(input_array).reshape((len(input_array), 1))
        self.output = self.activator.forward(np.dot(self.W, np.array(input_array).reshape((len(input_array), 1))) + self.b)

    def backward(self, delta_array):
        '''
        反向传播
        :param delta_array: 上一层传来的误差
        :return:
        '''
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

class SigmoidAcitvator():
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

class Network():
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i + 1], SigmoidAcitvator()))

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, train_label, train_data, rate):
        '''
        训练
        :param train_label: 训练集标签
        :param train_data: 训练集特征
        :param rate：学习速率
        '''
        for d in range(len(train_data)):
            self.predict(train_data[d])
            self.calc_gradient(train_label[d])
            self.update_weight(rate)

    def calc_gradient(self, label):
        predict = self.layers[-1].output
        delta = self.layers[-1].activator.backward(predict) * (np.array(label).reshape((len(label), 1)) - predict)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

def load_data():
    train_data = ImageLoader('train-images.idx3-ubyte', 60000).load()
    train_label = LabelLoader('train-labels.idx1-ubyte', 60000).load()
    test_data = ImageLoader('t10k-images.idx3-ubyte', 10000).load()
    test_label = LabelLoader('t10k-labels.idx1-ubyte', 10000).load()
    return train_data, train_label, test_data, test_label

def train_and_predict(network, train_data, train_label, test_data, test_label):
    epoch = 0
    last_error_ratio = 1.0
    while True:
        epoch += 1
        network.train(train_label, train_data, 0.3)
        print(epoch)
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data, test_label)
            print("epoch:"+str(epoch)+" error ratio:"+str(error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio
    num_test_data = len(test_data)
    predict_labels = []
    for i in range(num_test_data):
        predict = get_result(network.predict(test_data[i]))
        predict_labels.append(predict)
    return np.array(predict_labels)

def evaluate(network, test_data, test_label):
    error = 0
    num_test_data = len(test_data)
    for i in range(num_test_data):
        label = get_result(test_label[i])
        predict = get_result(network.predict(test_data[i]))
        if label != predict:
            error += 1
    return float(error) / float(num_test_data)

def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_data()
    network = Network([784, 300, 10])
    predict_labels = train_and_predict(network, train_data, train_label, test_data, test_label)
    print(predict_labels)
    pd.DataFrame(predict_labels).to_csv('submission.csv', index=False, encoding='utf8', header=False)