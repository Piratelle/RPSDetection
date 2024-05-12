from mmpose.apis import MMPoseInferencer

img_path = 'demo.jpg'

inferencer = MMPoseInferencer('hand')

result_generator = inferencer(img_path, show=True, out_dir='out')
result = next(result_generator)