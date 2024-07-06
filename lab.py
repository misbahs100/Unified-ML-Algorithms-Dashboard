from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)



# main app route
@app.route('/')
def main():
    return render_template('home.html')

@app.route('/home', methods=['POST'])
def home():
    return render_template('after.html')

@app.route('/lab-1')
def home1():
    return render_template('home1.html')

@app.route('/lab-7')
def home7():
    return render_template('home7.html')

@app.route('/lab-8')
def home8():
    return render_template('home8.html')

@app.route('/lab-9')
def home9():
    return render_template('home9.html')

@app.route('/lab-10')
def home10():
    return render_template('home10.html')

@app.route('/predict', methods=['POST'])
def lab1():
    shape = int(request.form['shape'])
    texture = int(request.form['texture'])
    weight = int(request.form['weight'])

    input_data = np.array([shape, texture, weight])

    # Training data
    training_data = np.array([
        [1, 1, 1],    # Apple
        [-1, -1, -1], # Orange
        [1, -1, 1],   # Apple
        [-1, 1, -1],  # Orange
    ])

    # Initialize weights and bias
    weights = np.random.rand(3)
    bias = np.random.rand()

    # Learning rate
    learning_rate = 0.1


    def perceptron(input_vector):
        activation = np.dot(input_vector, weights) + bias
        return 1 if activation >= 0 else -1

    def train_perceptron(training_data, targets, epochs, weights, bias):
        for _ in range(epochs):
            for i in range(len(training_data)):
                prediction = perceptron(training_data[i])
                error = targets[i] - prediction
                weights += learning_rate * error * training_data[i]
                bias += learning_rate * error

    train_perceptron(training_data, targets=np.array([1, -1, 1, -1]), epochs=100, weights=weights, bias=bias)
    prediction = perceptron(input_data)
    fruit = "Apple" if prediction == 1 else "Orange"

    return render_template('after1.html', shape=shape, texture=texture, weight=weight, fruit=fruit)





@app.route('/clusters-for-two-vars', methods=['POST'])
def lab7():
    
    sub1a = request.form['1a']
    sub1a = float(sub1a)
    sub1b = request.form['1b']
    sub1b = float(sub1b)

    sub2a = request.form['2a']
    sub2a = float(sub2a)
    sub2b = request.form['2b']
    sub2b = float(sub2b)

    sub3a = request.form['3a']
    sub3a = float(sub3a)
    sub3b = request.form['3b']
    sub3b = float(sub3b)

    sub4a = request.form['4a']
    sub4a = float(sub4a)
    sub4b = request.form['4b']
    sub4b = float(sub4b)

    sub5a = request.form['5a']
    sub5a = float(sub5a)
    sub5b = request.form['5b']
    sub5b = float(sub5b)

    sub6a = request.form['6a']
    sub6a = float(sub6a)
    sub6b = request.form['6b']
    sub6b = float(sub6b)

    sub7a = request.form['7a']
    sub7a = float(sub7a)
    sub7b = request.form['7b']
    sub7b = float(sub7b)

    cluster_num = request.form['cluster-num']
    cluster_num = int(cluster_num)

    # Given data
    # data = np.array([[sub1a, sub1b],
    #                 [sub2a, sub2b],
    #                 [sub3a, sub3b],
    #                 [sub4a, sub4b],
    #                 [sub5a, sub5b],
    #                 [sub6a, sub6b],
    #                 [sub7a, sub7b]])
    data = np.array([[1.0, 1.0],
                [1.5, 2.0],
                [3.0, 4.0],
                [5.0, 7.0],
                [3.5, 5.0],
                [4.5, 5.0],
                [3.5, 4.5]])

    # Number of clusters
    k = cluster_num

    # Initialize centroids randomly
    np.random.seed(0)
    centroids = data[np.random.choice(data.shape[0], size=k, replace=False)]
    print("centroids: ")
    print(centroids)


    def assign_to_clusters(data, centroids):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def update_centroids(data, cluster_assignments, k):
        new_centroids = np.zeros((k, data.shape[1]))
        for i in range(k):
            new_centroids[i] = np.mean(data[cluster_assignments == i], axis=0)
        return new_centroids

    # K-means algorithm
    max_iterations = 100
    for _ in range(max_iterations):
        cluster_assignments = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, cluster_assignments, k)

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    subjects = [f'Subject {i+1}' for i in range(len(data))]
    print(len(subjects))
    print(cluster_assignments)

    # Create a plot to visualize the clusters
    plt.figure(figsize=(8, 6))
    for cluster in range(k):
        cluster_data = np.array([data[i] for i in range(len(data)) if cluster_assignments[i] == cluster])
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster + 1}')

    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', label='Centroids')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title(f'K-Means Clustering (k={k})')
    plt.legend()
    
    # Save the plot to a file for displaying in the HTML page
    plot_filename = 'static/cluster_plot_lab7.png'
    plt.savefig(plot_filename)

     # Initialize empty lists for clusters
    clusters = [[] for _ in range(k)]
    cluster_indices = list(range(1, k + 1))

    for i in range(len(data)):
        cluster_index = cluster_assignments[i]
        clusters[cluster_index].append(f'Subject {i + 1}')

    cluster1 = [subjects[i] for i in range(len(data)) if cluster_assignments[i] == 0]
    cluster2 = [subjects[i] for i in range(len(data)) if cluster_assignments[i] == 1]
    return render_template('after7.html', clusters=clusters, cluster_indices=cluster_indices, cluster1=cluster1, cluster2=cluster2, k=k, plot_filename=plot_filename)



@app.route('/clusters-for-one-var', methods=['POST'])
def lab8():
    k = int(request.form['k'])
    
    stu1 = request.form['a']
    stu1 = float(stu1)
    stu2 = request.form['b']
    stu2 = float(stu2)
    stu3 = request.form['b']
    stu3 = float(stu3)
    stu4 = request.form['d']
    stu4 = float(stu4)
    stu5 = request.form['e']
    stu5 = float(stu5)
    stu6 = request.form['f']
    stu6 = float(stu6)

    # Given data
    students = ['A', 'B', 'C', 'D', 'E', 'F']
    cgpa = np.array([stu1, stu2, stu3, stu4, stu5, stu6])

    # Initial cluster centers
    # initial_centers = np.array([3.45, 2.98, 4.0])
    # Randomly select k centroids from the cgpa array
    np.random.seed(0)  # For reproducibility
    centroid_indices = np.random.choice(len(cgpa), k, replace=False)
    initial_centers = cgpa[centroid_indices]

    def assign_to_clusters(cgpa, centers):
        distances = np.abs(cgpa - centers[:, np.newaxis])
        cluster_assignments = np.argmin(distances, axis=0)
        return cluster_assignments

    def update_centers(cgpa, cluster_assignments, num_clusters):
        new_centers = np.zeros(num_clusters)
        for i in range(num_clusters):
            new_centers[i] = np.mean(cgpa[cluster_assignments == i])
        return new_centers

    # K-means algorithm
    max_iterations = 100
    # num_clusters = len(initial_centers)
    num_clusters = k
    centers = initial_centers.copy()

    for _ in range(max_iterations):
        cluster_assignments = assign_to_clusters(cgpa, centers)
        new_centers = update_centers(cgpa, cluster_assignments, num_clusters)

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    # Print final cluster assignments
    final_clusters = {}
    for i in range(num_clusters):
        final_clusters[i] = cgpa[cluster_assignments == i]

    for cluster, cgpa_values in final_clusters.items():
        student_names = [students[j] for j in range(len(students)) if cluster_assignments[j] == cluster]
        print(f"Cluster {cluster + 1}: Students {student_names} - CGPA {cgpa_values}")


    return render_template('after8.html', num_clusters=num_clusters, clusters=final_clusters, cluster_assignments=cluster_assignments, students=students)


@app.route('/find-distance', methods=['POST'])
def lab9():
    string1 = request.form['s1']
    string2 = request.form['s2']

    def levenshtein_distance(str1, str2):
        len_str1 = len(str1)
        len_str2 = len(str2)
        # Initialize a matrix to store distances
        distances = np.zeros((len_str1 + 1, len_str2 + 1), dtype=int)

        # Initialize the first row and column
        for i in range(len_str1 + 1):
            distances[i, 0] = i
        for j in range(len_str2 + 1):
            distances[0, j] = j

        # Compute distances using dynamic programming
        for i in range(1, len_str1 + 1):
            for j in range(1, len_str2 + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                distances[i, j] = min(
                    distances[i - 1, j] + 1,       # Deletion
                    distances[i, j - 1] + 1,       # Insertion
                    distances[i - 1, j - 1] + cost  # Substitution
                )
        distance = distances[len(string1)][len(string2)]
        print(distance)
        # Save the visualization as an image file
        plt.figure(figsize=(len_str2, len_str1))
        plt.imshow(distances, cmap='viridis', origin='upper')
        # Add labels and annotations for directions (The same code as before...)

        # Save the plot as an image file
        img_path = 'static/levenshtein_distance.png'
        plt.savefig(img_path, format='png')
        
        return img_path, distance
    
    img_path, distance = levenshtein_distance(string1, string2)
    return render_template('after9.html', string1=string1, string2=string2, distance=distance, img_path=img_path)

@app.route('/predict-lab-10', methods=['POST'])
def lab10():
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))

    def manhattan_distance(point1, point2):
        return np.sum(np.abs(point1 - point2))

    def knn_predict(training_data, labels, new_data, k, distance_metric):
        distances = []
        for i in range(len(training_data)):
            distance = distance_metric(new_data, training_data[i])
            distances.append((distance, labels[i]))
        
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        size_counts = {'S': 0, 'M': 0, 'L': 0}

        for distance, label in neighbors:
            size_counts[label] += 1

        predicted_size = max(size_counts, key=size_counts.get)
        return predicted_size

    training_data = np.array([
        [158, 58], [158, 59], [158, 63], [160, 59], [160, 60], [163, 60], 
        [163, 61], [160, 64], [163, 64], [165, 61], [165, 62], [165, 65], 
        [168, 62], [168, 63], [168, 66], [170, 63], [170, 64], [170, 68]
    ])
    labels = ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']

    height = int(request.form['height'])
    weight = int(request.form['weight'])
    k = int(request.form['k'])
    distance_metric = euclidean_distance if request.form['distance'] == 'euclidean' else manhattan_distance

    new_data = np.array([height, weight])
    predicted_size = knn_predict(training_data, labels, new_data, k, distance_metric)

    return render_template('after10.html', height=height, weight=weight, k=k, distance_metric=request.form['distance'], predicted_size=predicted_size)



if __name__ == "__main__":
    app.run(debug=True)
