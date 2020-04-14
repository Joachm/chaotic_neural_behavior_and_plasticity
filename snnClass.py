import tensorflow as tf
import numpy as np

class snnLayer:

    def __init__(self,name,  C, size, mode,rest, thres, peak, leak, a, b, c, d):
        self.name = name
        self.C = C
        self.size = size
        self.mode = mode
        self.rest = rest
        self.thres = thres
        self.peak = peak
        self.leak = leak
        self.a = a
        self.b = b
        self.c = tf.fill([self.size,1], c)
        self.d = tf.fill([self.size, 1],d)
    
        startVal = tf.constant_initializer(self.rest)
        with tf.name_scope(self.name):
            self.neurons = tf.get_variable(name=self.name+"Neurons", dtype=tf.float32, shape=[self.size, 1], initializer=startVal)
            self.recovery = tf.get_variable(name=self.name+"Recovery", dtype=tf.float32, shape=[self.size, 1], initializer=tf.zeros_initializer)
            self.connections = {}
            self.active = tf.get_variable(name=self.name+"Active", dtype=tf.float32, shape=[self.size, 1], initializer=tf.zeros_initializer)
            self.prevActive = tf.get_variable(name=self.name+"PrevActive", dtype=tf.float32, shape=[self.size, 1], initializer=tf.zeros_initializer)

    

    def addUniCon(self, layerName, layerShape, default, pct, lower, upper):
        weightMatrix = np.full((self.size, layerShape), default, dtype='float32')
        connect = np.random.uniform(lower, upper, weightMatrix.shape)
	
        mask = np.random.choice([0,1], size=weightMatrix.shape, p=[1-pct, pct]).astype(np.bool)

        weightMatrix[mask]= connect[mask]
        weightMatrix = tf.get_variable(name=self.name+layerName, dtype=tf.float32, initializer=weightMatrix)

        self.connections[layerName] = weightMatrix

    def addRecUniCon(self, default, pct, lower, upper):
        weightMatrix = np.full((self.size, self.size), default, dtype='float32')
        connect = np.random.uniform(lower, upper, weightMatrix.shape)

        mask = np.random.choice([0,1], size=weightMatrix.shape, p=[1-pct, pct]).astype(np.bool)
        
        weightMatrix[mask]= connect[mask]    
        np.fill_diagonal(weightMatrix, 0.)

        weightMatrix = tf.get_variable(name=self.name+'recurrent', dtype=tf.float32, initializer=weightMatrix)

        self.connections['recurrent']=weightMatrix
    
    def shutdown(self):
        self.neurons = tf.assign(self.neurons, self.neurons * 0 + self.rest)
        self.recovery = tf.assign(self.recovery,self.recovery * 0)
        self.active = tf.assign(self.active, self.active * 0)
        self.prevActive = tf.assign(self.prevActive,self.prevActive *  0)

    def updateNeurons(self, inp, step):

        self.neurons = tf.assign(self.neurons, self.neurons + step*(((self.leak* (self.neurons - self.rest)*(self.neurons-self.thres)) - self.recovery  + inp)/self.C))

        self.recovery = tf.assign(self.recovery, self.recovery + step*(self.a * (self.b * (self.neurons - self.rest)-self.recovery )))

        active = tf.greater_equal(self.neurons, self.peak)
        active = tf.reshape(tf.cast(active, tf.float32), [self.size,1])
        
        self.active = tf.assign(self.active, active)


    def sendOutput(self, layerName, maxStrength):
        summedSend = tf.matmul(tf.transpose(self.active), self.connections[layerName], a_is_sparse=True,b_is_sparse=True ) * maxStrength * self.mode
        summedSend = tf.clip_by_value(summedSend, -400, 2000)
        summedSend = tf.reshape(summedSend, [tf.shape(self.connections[layerName])[1],1])
        return summedSend

    def sendOutput2(self, layerName, maxStrength):
        summedSend = tf.matmul(tf.transpose(self.active), self.connections[layerName], a_is_sparse=True,b_is_sparse=True ) * maxStrength * self.mode
        summedSend = tf.clip_by_value(summedSend, -300, 300)
        summedSend = tf.reshape(summedSend, [tf.shape(self.connections[layerName])[1],1])
        return summedSend
   
    def sendOutput3(self, layerName, maxStrength):
        summedSend = tf.matmul(tf.transpose(self.active), self.connections[layerName], a_is_sparse=True,b_is_sparse=True ) * maxStrength * self.mode
        summedSend = tf.clip_by_value(summedSend, -450, 300)
        summedSend = tf.reshape(summedSend, [tf.shape(self.connections[layerName])[1],1])
        return summedSend
   


    def reset(self):
        restore = self.active * self.c        
        inactive = tf.cast(tf.equal(self.active, 0), tf.float32)

        self.neurons = tf.assign( self.neurons, self.neurons * inactive)
        self.neurons = tf.assign( self.neurons, self.neurons + restore)

        recRestore = self.active * self.d
        self.recovery = tf.assign(self.recovery, self.recovery + recRestore)

        self.prevActive= tf.assign(self.prevActive, self.active)

    def STDP(self, layerName, receiveLayerActive, posLR, negLR):

        send = self.connections[layerName] * self.active
        sendTransp = tf.transpose(send)
        posLearningWeights = sendTransp * receiveLayerActive
        posLearningWeights = tf.transpose(posLearningWeights)

        potentiationChange = tf.pow(posLearningWeights, posLR) - posLearningWeights

        self.connections[layerName] = tf.assign(self.connections[layerName], self.connections[layerName] + potentiationChange)
        

        recLayerInactive = tf.cast(tf.equal(receiveLayerActive, 0), tf.float32)
        negLearningWeights = sendTransp * recLayerInactive
        negLearningWeights = tf.transpose(negLearningWeights)
        depressionChange = tf.pow(negLearningWeights, negLR) - negLearningWeights
        
        self.connections[layerName] =  tf.assign(self.connections[layerName], self.connections[layerName] + depressionChange)

    
    def iSTDP(self, layerName, receiveLayerActive, posLR, negLR):

        send = self.connections[layerName] * self.active
        sendTransp = tf.transpose(send)
        negLearningWeights = sendTransp * receiveLayerActive
        negLearningWeights = tf.transpose(negLearningWeights)

        depressionChange = tf.pow(negLearningWeights, negLR) - negLearningWeights

        self.connections[layerName] =  tf.assign(self.connections[layerName], self.connections[layerName] + depressionChange)



        recLayerInactive = tf.cast(tf.equal(receiveLayerActive, 0), tf.float32)
        posLearningWeights = sendTransp * recLayerInactive
        posLearningWeights = tf.transpose(posLearningWeights)
        potentiationChange = tf.pow(posLearningWeights, posLR) - posLearningWeights

        
        self.connections[layerName] = tf.assign(self.connections[layerName], self.connections[layerName] + potentiationChange)
        


    def recurrentSTDP(self, posLR, negLR):

        allActive = self.prevActive + self.active
        active = tf.greater_equal(allActive, 1)
        allActive = tf.reshape(tf.cast(active, tf.float32), [self.size,1])

        send = self.connections['recurrent'] * allActive
        sendTransp = tf.transpose(send)
        posLearningWeights = sendTransp * self.active

        posLearningWeights = tf.transpose(posLearningWeights)

        power = tf.pow(posLearningWeights,posLR) 

        potentiationChange = power - posLearningWeights

        self.connections['recurrent'] = tf.assign(self.connections['recurrent'], self.connections['recurrent'] + potentiationChange)

        inactive = tf.cast(tf.equal(self.active, 0), tf.float32)
        negLearningWeights = sendTransp * inactive
        negLearningWeights = tf.transpose(negLearningWeights)
        depressionChange = tf.pow(negLearningWeights, negLR) - negLearningWeights


        self.connections['recurrent'] =  tf.assign(self.connections['recurrent'], self.connections['recurrent'] + depressionChange)




    def recurrentiSTDP(self, posLR, negLR):

        allActive = self.prevActive + self.active

        send = self.connections['recurrent'] * allActive
        sendTransp = tf.transpose(send)
        negLearningWeights = sendTransp * self.active
        negLearningWeights = tf.transpose(negLearningWeights)

        depressionChange = negLearningWeights** negLR - negLearningWeights

        self.connections['recurrent'] =  tf.assign(self.connections['recurrent'], self.connections['recurrent'] + depressionChange)


        inactive = tf.cast(tf.equal(self.active, 0), tf.float32)
        posLearningWeights = sendTransp * inactive
        posLearningWeights = tf.transpose(posLearningWeights)
        potentiationChange = tf.pow(posLearningWeights, posLR) - posLearningWeights

        self.connections['recurrent'] = tf.assign(self.connections['recurrent'], self.connections['recurrent'] + potentiationChange)





