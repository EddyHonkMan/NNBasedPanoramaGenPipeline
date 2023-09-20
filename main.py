import videoloader as vl

if __name__ == '__main__':
    FILE_NAME = 'videos/Stairs1s.mp4'
    video = vl.VideoLoader(FILE_NAME)
    video.foreground_background_separate(24, 2)
    vl.show_video(video.frames)
    vl.show_video(video.foregrounds)

