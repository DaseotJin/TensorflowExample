# Lab 2 Linear Regression
import tensorflow as tf #Define 
tf.set_random_seed(777)  # for reproducibility 랜덤 시드 생성 시드 인자에 따라 결과값이 달라짐

# Try to find values for W and b to compute Y = W * X + b
W = tf.Variable(tf.random_normal([1]), name="weight") # W에 대한 변수 선언 (위에 선언된 시드에 의해)
b = tf.Variable(tf.random_normal([1]), name="bias") # b에 대한 변수 선언 (그릇만, 위도) 세션 안에서는 안됨

# placeholders for a tensor that will be always fed using feed_dict
# See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32, shape=[None]) # 파이썬 안에서 접근 가능
Y = tf.placeholder(tf.float32, shape=[None]) # 그릇 만들기

# Our hypothesis is X * W + b
hypothesis = X * W + b #가상함수수식

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y)) # reduce_mean = 평균값, square는 제곱

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) #learning_rate 그래프에서 이동하는 수치
# train 변수 : 위에서 설정한 cost 변수를 줄여주는 작업

# Launch the graph in a session.
with tf.Session() as sess: # 세션 생성
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer()) #세션을 initiallizer로 실행

    # Fit the line
    for step in range(2001): #학습단
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
        # 각 변수에 선언한 값을 대입, feed_dict라는 메소드에 인풋과 아웃풋을 선언
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val) # step을 20으로 나눴을때 나머지가 0일때 print안의 인자 값를 출력

            #2000번 학습 이후 결정된 W와 b값을 정함? feed_dict에 x와 y값을 각각 대입해서 W와 b값을 결정
          

    # Testing our model
    print(sess.run(hypothesis, feed_dict={X: [5]})) # feed_dict에 X값 대입, 가상 함수 러닝 값을 출력
    print(sess.run(hypothesis, feed_dict={X: [2.5]})) # feed_dict에 X값 대입, 가상 함수 러닝 값을 출력
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]})) # feed_dict에 X값 대입, 가상 함수 러닝 값을 출력

    # Learns best fit W:[ 1.],  b:[ 0]
    """
    0 3.5240757 [2.2086694] [-0.8204183]
    20 0.19749963 [1.5425726] [-1.0498911]
    ...
    1980 1.3360998e-05 [1.0042454] [-0.00965055]
    2000 1.21343355e-05 [1.0040458] [-0.00919707]
    [5.0110054]
    [2.500915]
    [1.4968792 3.5049512]
    """

    # Fit the line with new training data
    for step in range(2001): #포문으로 2000번 step 돌림
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
            # 위에서 학습한 결과로 다시 한번 결과 도출

    # Testing our model
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

    # Learns best fit W:[ 1.],  b:[ 1.1]
    """
    0 1.2035878 [1.0040361] [-0.00917497]
    20 0.16904518 [1.2656431] [0.13599995]
    ...
    1980 2.9042917e-07 [1.00035] [1.0987366]
    2000 2.5372992e-07 [1.0003271] [1.0988194]
    [6.1004534]
    [3.5996385]
    [2.5993123 4.599964 ]
    """