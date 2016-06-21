all: drone.mp4 vo

affine: latchAff.o bitMatcher.o
	g++ -std=c++11 `pkg-config --cflags opencv` affTest.cpp latchAff.o bitMatcher.o -I/usr/local/cuda-7.5/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core -o affTest

vo: latch.o bitMatcher.o #gpuFacade.o #fast.o
	g++ -std=c++11 `pkg-config --cflags opencv` vo.cpp latch.o bitMatcher.o -I/usr/local/cuda-7.5/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core -o vo

vo2: latch.o bitMatcher.o gpuFacade.o   #fast.o
	g++ `pkg-config --cflags opencv` vo2.cpp latch.o bitMatcher.o -I/usr/local/cuda-7.5/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core -o vo2

demo2: drone.mp4 vo2
	./vo2 drone.mp4 620 | perl ./driveGnuPlotStreams.pl 12 4 200 4200 200 200 0 0 0 0 0 0 0 0 950x200+960+30 950x200+960+780 950x200+960+280 950x200+960+530 'pitch' 'yaw' 'roll' 'polar translation angle' 'azimuthal translation angle' 'z' 'keypoints' 'matches' '100 * threshold' 'cpu [ms]' 'gpu [ms]' 'defect' 0 0 0 1 1 1 2 2 2 3 3 1

demo2_no_gnuplot: drone.mp4 vo2
	./vo2 drone.mp4 620

demo: drone.mp4 vo
	./vo drone.mp4 620 | perl ./driveGnuPlotStreams.pl 12 4 200 4200 200 200 0 0 0 0 0 0 0 0 950x200+960+30 950x200+960+780 950x200+960+280 950x200+960+530 'pitch' 'yaw' 'roll' 'polar translation angle' 'azimuthal translation angle' 'z' 'keypoints' 'matches' '100 * threshold' 'cpu [ms]' 'gpu [ms]' 'defect' 0 0 0 1 1 1 2 2 2 3 3 1
#620
#4400
demo_no_gnuplot: drone.mp4 vo
	./vo drone.mp4 620

drone.mp4:
	youtube-dl -f 137 https://www.youtube.com/watch?v=wneCezU_VQ4
	mv Raw\ FPV\ Training\ Session\ -\ Dirt\ Bike\ Visit\ in\ Park-wneCezU_VQ4.mp4 drone.mp4

gpuFacade.o: latch.o bitMatcher.o
	g++ `pkg-config --cflags opencv` -c gpuFacade.cpp latch.o bitMatcher.o -I/usr/local/cuda-7.5/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core


#fast.o:
#	nvcc -c -lineinfo -O3 -o fast.o       fast.cu       -gencode arch=compute_52,code=sm_52 -I/home/chris/cub-1.5.2/

latch.o:
	nvcc -c -lineinfo -Xptxas -v -use_fast_math -O3 -o latch.o      latch.cu      -gencode arch=compute_52,code=sm_52

latchAff.o:
	nvcc -c -lineinfo -Xptxas -v -use_fast_math -O3 -o latchAff.o      latchAff.cu      -gencode arch=compute_52,code=sm_52


min: latch.o bitMatcher.o
	g++ `pkg-config --cflags opencv` min.cpp latch.o bitMatcher.o -I/usr/local/cuda-7.5/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn -lopencv_dpm -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core -o min
	./min ob/1.png ob/2.png

bitMatcher.o:
	nvcc -c -lineinfo -Xptxas -v -use_fast_math -O3 -o bitMatcher.o bitMatcher.cu -gencode arch=compute_52,code=sm_52

clean:
	rm vo; rm latch.o; rm bitMatcher.o; rm gpuFacade.o; rm latchAff.o; rm vo2; rm affTest;

run: vo
	./vo

#plot: vo
#	./vo $(video) $(skip) $(w) $(h) | feedgnuplot --stream 0.01 --lines --nopoints --legend 0 pitch --legend 1 yaw --legend 2 roll --xlen 200
