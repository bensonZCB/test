#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <stdint.h>
#include "yolo_post_process.h"
#include "yolo_rknn_run.h"
#include "rknn_api.h"

model_type = YOLOV5;
int num_labels = 1;
int yolo_anchors[3][6] = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326}};

inline static int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, float *outputLocations, int *class_id, int *order, float threshold, bool class_agnostic)
{
    // printf("class_agnostic: %d\n", class_agnostic);
    for (int i = 0; i < validCount; ++i)
    {
        if (order[i] == -1)
        {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1)
            {
                continue;
            }

            if (class_agnostic == false && class_id[n] != class_id[m]){
                continue;
            }

            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(
    float *input,
    int left,
    int right,
    int *indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

// static float sigmoid(float x)
// {
//     return 1.0 / (1.0 + expf(-x));
// }

// static float unsigmoid(float y)
// {
//     return -1.0 * logf((1.0 / y) - 1.0);
// }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

static int process_i8(int8_t *input, int *anchor, int anchor_per_branch, int grid_h, int grid_w, int height, int width, int stride,
                   float *boxes, float *boxScores, int *classId,
                   float threshold, int32_t zp, float scale, MODEL_TYPE yolo, int index)
{
    const int PROP_BOX_SIZE = 5 + num_labels;

    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres = threshold;
    auto thres_i8 = qnt_f32_to_affine(thres, zp, scale);
    // puts("==================================");
    // printf("threash %f\n", thres);
    // printf("thres_i8 %u\n", thres_i8);
    // printf("scale %f\n", scale);
    // printf("zp %d\n", zp);
    // puts("==================================");

    //printf("it goes here: file %s, at line %d\n", __FILE__, __LINE__);
    for (int a = 0; a < anchor_per_branch; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {

            for (int j = 0; j < grid_w; j++)
            {

                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                //printf("The box confidence in i8: %d\n", box_confidence);
                if (box_confidence >= thres_i8)
                {
                    // printf("box_conf %u, thres_i8 %u\n", box_confidence, thres_i8);
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < num_labels; ++k)
                    {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }

                    float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
                    float class_prob_f32 = deqnt_affine_to_f32(maxClassProbs, zp, scale);
                    float limit_score = 0;
                    if (yolo == YOLOX){
                        limit_score = class_prob_f32;
                    }
                    else{
                        limit_score = box_conf_f32* class_prob_f32;
                    }
                    //printf("limit score: %f\n", limit_score);
                    if (limit_score > CONF_THRESHOLD){
                        float box_x, box_y, box_w, box_h;
                        if(yolo == YOLOX){
                            box_x = deqnt_affine_to_f32(*in_ptr, zp, scale);
                            box_y = deqnt_affine_to_f32(in_ptr[grid_len], zp, scale);
                            box_w = deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale);
                            box_h = deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale);
                            box_w = exp(box_w)* stride;
                            box_h = exp(box_h)* stride;
                        }   
                        else{
                            box_x = deqnt_affine_to_f32(*in_ptr, zp, scale) * 2.0 - 0.5;
                            box_y = deqnt_affine_to_f32(in_ptr[grid_len], zp, scale) * 2.0 - 0.5;
                            box_w = deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale) * 2.0;
                            box_h = deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale) * 2.0;
                            box_w = box_w * box_w;
                            box_h = box_h * box_h;
                        }
                        box_x = (box_x + j) * (float)stride;
                        box_y = (box_y + i) * (float)stride;
                        box_w *= (float)anchor[a * 2];
                        box_h *= (float)anchor[a * 2 + 1];
                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);

                        // boxes.push_back(box_x);
                        // boxes.push_back(box_y);
                        // boxes.push_back(box_w);
                        // boxes.push_back(box_h);
                        // boxScores.push_back(box_conf_f32* class_prob_f32);
                        // classId.push_back(maxClassId);
                        boxes[index + 4*validCount + 0] = box_x;
                        boxes[index + 4*validCount + 1] = box_y;
                        boxes[index + 4*validCount + 2] = box_w;
                        boxes[index + 4*validCount + 3] = box_h;
                        boxScores[index + validCount] = (box_conf_f32* class_prob_f32);
                        classId[index + validCount] = maxClassId;

                        validCount++;
                        if (validCount >= VALID_COUNT_MAX)
                        {
                            printf("break: validCount >= VALID_COUNT_MAX\n");
                            return validCount;
                        }
                    }
                }
            }
        }
    }
    return validCount;
}



// static int process_fp(float *input, int *anchor, int anchor_per_branch,int grid_h, int grid_w, int height, int width, int stride,
//                    std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId,
//                    float threshold, MODEL_TYPE yolo)
// {

//     const int PROP_BOX_SIZE = 5 + num_labels;

//     int validCount = 0;
//     int grid_len = grid_h * grid_w;
//     // float thres_sigmoid = threshold;
//     for (int a = 0; a < anchor_per_branch; a++)
//     {
//         for (int i = 0; i < grid_h; i++)
//         {
//             for (int j = 0; j < grid_w; j++)
//             {
//                 float box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
//                 if (box_confidence >= threshold)
//                 {
//                     int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
//                     float *in_ptr = input + offset;

//                     float maxClassProbs = in_ptr[5 * grid_len];
//                     int maxClassId = 0;
//                     for (int k = 1; k < num_labels; ++k)
//                     {
//                         float prob = in_ptr[(5 + k) * grid_len];
//                         if (prob > maxClassProbs)
//                         {
//                             maxClassId = k;
//                             maxClassProbs = prob;
//                         }
//                     }
//                     float box_conf_f32 = (box_confidence);
//                     float class_prob_f32 = (maxClassProbs);
//                     float limit_score = 0;
//                     if (yolo == YOLOX){
//                         limit_score = class_prob_f32;
//                     }
//                     else{
//                         limit_score = box_conf_f32* class_prob_f32;
//                     }
//                     // printf("limit score: %f", limit_score);
//                     if (limit_score > CONF_THRESHOLD){
//                         float box_x, box_y, box_w, box_h;
//                         if (yolo == YOLOX){
//                             box_x = *in_ptr;
//                             box_y = (in_ptr[grid_len]);
//                             box_w = exp(in_ptr[2* grid_len])* stride;
//                             box_h = exp(in_ptr[3* grid_len])* stride;
//                         }
//                         else{
//                             box_x = *in_ptr * 2.0 - 0.5;
//                             box_y = (in_ptr[grid_len]) * 2.0 - 0.5;
//                             box_w = (in_ptr[2 * grid_len]) * 2.0;
//                             box_h = (in_ptr[3 * grid_len]) * 2.0;
//                             box_w *= box_w;
//                             box_h *= box_h;
//                         }
//                         box_x = (box_x + j) * (float)stride;
//                         box_y = (box_y + i) * (float)stride;
//                         box_w *= (float)anchor[a * 2];
//                         box_h *= (float)anchor[a * 2 + 1];
//                         box_x -= (box_w / 2.0);
//                         box_y -= (box_h / 2.0);
                        
//                         boxes.push_back(box_x);
//                         boxes.push_back(box_y);
//                         boxes.push_back(box_w);
//                         boxes.push_back(box_h);
//                         boxScores.push_back(box_conf_f32* class_prob_f32);
//                         classId.push_back(maxClassId);
//                         validCount++;
//                     }
//                 }
//             }
//         }
//     }
//     return validCount;
// }

static float *filterBoxes = NULL;
static float *boxesScore = NULL;
static int *classId = NULL;
static int *indexArray = NULL;

int post_process_init(void)
{
    if (filterBoxes == NULL || boxesScore == NULL || classId == NULL || indexArray == NULL)
    {
        free(filterBoxes);
        free(boxesScore);
        free(classId);
        free(indexArray);

        filterBoxes = malloc(4*VALID_COUNT_MAX*sizeof(float));
        boxesScore = malloc(VALID_COUNT_MAX*sizeof(float));
        classId = malloc(VALID_COUNT_MAX*sizeof(int));
        indexArray = malloc(VALID_COUNT_MAX*sizeof(int));
        if (filterBoxes == NULL || boxesScore == NULL || classId == NULL)
        {
            return -1;
        }
    }

    return 0;
}

int post_process(void **rk_outputs, int width, int height, rknn_tensor_attr * out_attr, int output_cnt, detect_result_t *results)
{
    int result_count = 0;
    int validCount = 0;
    int strides[3] = {8,16,32};
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int tmp_cnt = 0;
    for (int i=0; i<output_cnt; i++){
        stride = strides[i];
        grid_h = height/ stride;
        grid_w = width/ stride;
        tmp_cnt = process_i8((int8_t*) rk_outputs[i], yolo_anchors[i], ANCHOR_PER_BRANCH, grid_h, grid_w, height, width, stride, 
                    filterBoxes, boxesScore, classId, CONF_THRESHOLD, out_attr[i].zp, out_attr[i].scale, model_type, validCount);
        validCount += tmp_cnt;
    }

    // no object detect
    if (validCount <= 0){
        return 0;
    }
    for (int i = 0; i < validCount; ++i){
        indexArray[i] = i;
    }

    quick_sort_indice_inverse(boxesScore, 0, validCount - 1, indexArray);

    if (model_type == YOLOV5 || model_type == YOLOV7){
        nms(validCount, filterBoxes, classId, indexArray, NMS_THRESHOLD, false);
    }
    else if (model_type == YOLOX){
        nms(validCount, filterBoxes, classId, indexArray, NMS_THRESHOLD, true);
    }
    
    /* box valid detect target */
    //std::vector<detect_result_t> results;
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || boxesScore[i] < CONF_THRESHOLD || result_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float prop = boxesScore[i];

        results[result_count].class_id = id;
        results[result_count].prop = prop*255;
        results[result_count].left = (int)(clamp(x1, 0, width-1));
        results[result_count].top = (int)(clamp(y1, 0, height-1));
        results[result_count].right = (int)(clamp(x2, 0, width-1));
        results[result_count].bottom = (int)(clamp(y2, 0, height-1));

        //printf("result (%4d, %4d, %4d, %4d) \n",results[result_count].left, results[result_count].top,results[result_count].right, results[result_count].bottom);
        result_count++;
    }

    return result_count;
}
