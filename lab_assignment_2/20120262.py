# import library
import numpy as np
import json
from map_feature import map_feature


class LogisticRegression():
    def __init__(self, alpha=0.5,lamda = 1, iters=10000):
        self.alpha = alpha
        self.iters = iters
        self.theta = None
        self.lamda = lamda
    def sigmoid(self,X,temp_theta):
        z=np.dot(X,temp_theta)
        g= 1./(1 + np.exp(-z))
        return np.array(g)
        
    # compute_cost: cal the cost of model of data set 
    def compute_cost(self,x, y, temp_theta):
        m = y.shape[0]
        hx = self.sigmoid(x,temp_theta)
        Jl = -np.sum( y * np.log(hx) + (1-y) * np.log(1-hx))/float(m)
        Jr= (self.lamda*np.sum(temp_theta**2))/(2*m)
        J = Jl + Jr
        J = np.squeeze(J)
        return J
    # compute_gradient: calculate the gradient vector of the cost function   
    def compute_gradient(self,x,y,temp_theta):
        m = y.shape[0]
        h_theta = self.sigmoid(x,temp_theta)

        dJ = np.dot((h_theta - y),x)/float(m)
        dJ[1:] += (self.lamda/float(m)) * temp_theta[1:]

        return dJ
        

    def gradient_descent(self,X,y):
        temp_theta = np.zeros(X.shape[1])
        for i in range(self.iters):
            dJ = self.compute_gradient(X,y,temp_theta)
            temp_theta = temp_theta - self.alpha*dJ
            cost = self.compute_cost(X, y, temp_theta)
            print('Iter: {} - cost = {}'.format(i+1, cost))
        return temp_theta
        
    def predict(self,X):
        h_theta = self.sigmoid(X, self.theta)
        y = (h_theta >= 0.5).astype(int)
        return y


    def fit(self,x, y):
        self.theta = self.gradient_descent(x,y)
        
    def calculate_accuracy(self,x, y): 
        pred = []
        for i in range(len(x)):
            pred.append(self.predict(x[i]))
        correct = 0

        for i in range(len(y)):
            if pred[i] == y[i]:
                correct += 1
        accuracy =  (correct/len(y))

        print('Caculate accuracy...DONE!')
        return accuracy
    
    def evaluate(self,y_test, y_pred,labels):
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_pred)))


        tp = sum((y_test == 1) & (y_pred == 1))
        fp = sum((y_test == 0) & (y_pred == 1))
        tn = sum((y_test == 0) & (y_pred == 0))
        fn = sum((y_test == 1) & (y_pred == 0))

    
        no_condition_support = len([i for i in y_test if i == 0])
        with_condition_support = len([i for i in y_test if i == 1])

        no_condition_precision = tn / (tn + fn)
        no_condition_recall = tn / (tn + fp)
        no_condition_f1_score = 2 * no_condition_precision * no_condition_recall / (no_condition_precision + no_condition_recall)

        with_condition_precision = tp / (tp + fp)
        with_condition_recall = tp / (tp + fn)
        with_condition_f1_score = 2 * with_condition_precision * with_condition_recall / (with_condition_precision + with_condition_recall)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        macro_precision = (no_condition_precision + with_condition_precision) / 2
        macro_recall = (no_condition_recall + with_condition_recall) / 2
        macro_f1_score = (no_condition_f1_score + with_condition_f1_score) / 2

        weighted_precision = (no_condition_support / len(y_test)) * no_condition_precision + (with_condition_support / len(y_test)) * with_condition_precision
        weighted_recall = (no_condition_support / len(y_test)) * no_condition_recall + (with_condition_support / len(y_test)) * with_condition_recall
        weighted_f1_score = (no_condition_support / len(y_test)) * no_condition_f1_score + (with_condition_support / len(y_test)) * with_condition_f1_score

        report = "\t\t\t\t\tprecision\trecall\tf1-score\tsupport\n"
        report += "\n{0}\t{1:.2f}\t\t {2:.2f}\t  {3:.2f}\t      {4:2}".format("without condition", no_condition_precision, no_condition_recall, no_condition_f1_score, no_condition_support)
        report += "\n{0}  \t{1:.2f}\t\t {2:.2f}\t  {3:.2f}\t      {4:2}".format("with condition", with_condition_precision, with_condition_recall, with_condition_f1_score, with_condition_support)
        report += "\n\naccuracy\t\t\t\t\t\t\t\t  {0:.2f}\t      {1}".format(accuracy, len(y_test))
        report += "\nmacro avg       \t{0:.2f}\t\t {1:.2f}\t  {2:.2f}\t      {3}".format(macro_precision, macro_recall, macro_f1_score, len(y_test))
        report += "\nweighted avg    \t{0:.2f}\t\t {1:.2f}\t  {2:.2f}\t      {3}".format(weighted_precision, weighted_recall, weighted_f1_score, len(y_test))
        no_con_array = np.array([no_condition_precision, no_condition_recall, no_condition_f1_score, no_condition_support])
        con_array = np.array([with_condition_precision, with_condition_recall, with_condition_f1_score, with_condition_support])
        accu_array = np.array([accuracy, len(y_test)])
        macro_array = np.array([macro_precision, macro_recall, macro_f1_score, len(y_test)])
        weighted_array = np.array([weighted_precision, weighted_recall, weighted_f1_score, len(y_test)])
        report_array = np.vstack((no_con_array,con_array,macro_array,weighted_array))
        return report,report_array,accu_array 

        

    def save_model(self): 
        model = {
            'theta': self.theta.tolist()
        }

        with open('model.json', 'w') as file:
            json.dump(model, file)
        
        print('Save model....DONE!')

    def print_predict_accuracy(self,x, y): 
        accuracy = self.calculate_accuracy(x, y)
        cost = self.compute_cost(x, y, self.theta)
        return accuracy, cost

    
    
    # Read the training configuration from file config.json
def read_training_configuration():
    #Load config    
    with open('config.json',) as f:
        configs = json.load(f)
    return configs['Alpha'], configs['Lambda'], configs['NumIter']


    # Read the training data from file training_data.txt
def read_training_data(): 
    datas = np.loadtxt('training_data.txt', delimiter = ',')
    
    x_raw = datas[: , 0:2]
    x1 = np.array([float(i) for i in datas[:,0]])
    x2 = np.array([float(i) for i in datas[:,1]])
    X0 = map_feature(x1,x2)
    y = datas[: , 2]
    
    return x_raw, X0, np.array(y)

def save_classification_report(report,accu_array,filename,labels):
    temp1 =labels[0]
    temp2 = labels[1]
    with open(filename, "w") as file:
        model = {
            temp1:{'precision':f'{report[0][0]:.2f}','recall':f'{report[0][1]:.2f}','f1-score':f'{report[0][2]:.2f}','support':f'{report[0][3]}'},
            temp2:{'precision':f'{report[1][0]:.2f}','recall':f'{report[1][1]:.2f}','f1-score':f'{report[1][2]:.2f}','support':f'{report[1][3]}'},
            'macro avg':{'precision':f'{report[2][0]:.2f}','recall':f'{report[2][1]:.2f}','f1-score':f'{report[2][2]:.2f}','support':f'{report[2][3]}'},
            'weighted avg':{'precision':f'{report[3][0]:.2f}','recall':f'{report[3][1]:.2f}','f1-score':f'{report[3][2]:.2f}','support':f'{report[3][3]}'},
            'accuracy':{'f1-score':f'{accu_array[0]:.2f}','support':f'{accu_array[1]}'}
        }
        json.dump(model, file)

def main():
    # - Read the training configuration from file config.json.
    x_raw, x, y= read_training_data()

    alpha, lamda, numiter = read_training_configuration()

    model = LogisticRegression(alpha=alpha,lamda=lamda,iters=numiter)
    
    model.fit(x, y)
    
    # - Save model to file model.json.
    
    model.save_model()
    y_pred = model.predict(x)
    # - Make prediction and calculate accuracy of training data set, save result to file accuracy.json.
    accuracy, cost = model.print_predict_accuracy(x, y)
    target_names = ['without condition', 'with condition']
    evaluation,evaluation_array,accu_array = model.evaluate(y,y_pred,labels=target_names)
    print(evaluation)
    save_classification_report(evaluation_array,accu_array, "classification_report.json",target_names)
    print('------------------------------')
    print('Cost function:', round(cost,3))
    print('Accuracy: {} %'. format(round(accuracy,3)))
    print('------------------------------')
    

if __name__ == "__main__":
    main()

