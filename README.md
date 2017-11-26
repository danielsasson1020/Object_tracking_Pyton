# Object_tracking_Pyton


- After loading the project
- Change the value of video_name according to the video you want to load
  and run the program.
- by default the program tracks multiple objects
- press "t" to enter single object tracking, select on the video the top-left and bottom-right points of the object to want to track
- press "c" to clear and return to multiple objects tracking
- Move the sliders as follows:

Threshold = Smaller values will lead to more motion being detected tho can lead to noisy tracking,
			larger values to less motion detected and less noise.

search range = To use with single object tracking, lower values help to track small and slow moving objects

Avg Weight = Smaller values will make the avg background update slower, making it easier to follow slow moving 		objects ,but will lead to more "noisy" tracking ,higher values will make the background update faster which helps tracking fast moving objects

Delay = determines how fast the video plays

- Press ESC to exit the program.
