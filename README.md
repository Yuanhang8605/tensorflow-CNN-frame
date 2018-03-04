# CNN_FRAME
This year I decide to quit my job in nuclear design institute, and switch to a new job in computer vision. It's a painful process, because I start from zero,learning python, c/c++, and tensorflow. I can insist because the strong interest. Good luck to me! 

In this repo, I learn tensorflow API by reading others' code, and modify some part to my style. some are written by myself. stand on the giant's shoulder! 

You'll know how to do:
(1) transfer image to tf example proto, TFRecord
(2) build image pipeline input to CNN using tf.data.Dataset API
(3) preprocess the images
(4) how to efficiently construct CNN using light weight high level API slim. 
    slim provide arg_scope to set the default parameters. 
    slim wrap low level API, provide layers concept. 
    the process of construct a CNN using slim:
    a) touch a file net_utils.py
      in this file, you should:
       define a arg_scope, define the default parameters;
      ```python
      def resnet_arg_scope(...):
        with slim.arg_scope(...)
          with slim.arg_scope(...)
            ... as arg_sc
        return arg_sc
      ```
       define your own ops, if you want to let your ops can be managed by slim.arg_scope, you can decorate it with @slim.add_arg_scope
    b) touch a file net.py
      in this file you should construct the main structure of your CNN net. 
    
(5) how to test your cnn using tf.test.TestCase API

..... coming soon. 

join me, learn tensorflow together! 