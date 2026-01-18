# WhatsApp controlled Angry Birds

We used a pygame implementation of angrybirds by marblexu and modified it to use bird noises sent via WhatsApp voice messages to select birds and fire them. This was for the 'unintended behaviour' theme of CamHacks 25.

We used Twilio to send the audio files from WhatsApp to our server. To analyse and classify the 'bird noises' we used a pretrained 14 layer CNN with a linear head at the end that we trained ourselves on synthetic data, achieveing >90% accuracy. This is pretty impressive given that the audios were human imitations of angry bird noises which made for a challenging classification task on minimal data and compute. We used spectrograms to calibrate the bird noises so that frequency determines shot angle and volume determines shot power.

Thanks to Aaron MacWhirter, Matyas Vecsei and William Goacher for their contributions to this project.
