#ifndef _YOLO_H__
#define _YOLO_H__

#define NMS_THRESHOLD 0.45
#define CONF_THRESHOLD 0.5 
#define VALID_COUNT_MAX 64
#define OBJ_NUMB_MAX_SIZE 8
#define ANCHOR_PER_BRANCH 3

typedef struct __detect_result_t
{
    int class_id;
    float prop;
    int left;
    int top;
    int right;
    int bottom;
} detect_result_t;

typedef enum{
    YOLOV5 = 0,
    YOLOX = 1,
    YOLOV7 = 2,
    MOBILENET = 2,
}MODEL_TYPE;
int post_process_init(void);
#endif