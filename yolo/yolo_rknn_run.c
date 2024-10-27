#include <rga/im2d.h>
#include <rga/rga.h>
#include <rk_comm_tde.h>
#include <rk_debug.h>
#include <rk_mpi_cal.h>
#include <rk_mpi_ivs.h>
#include <rk_mpi_mb.h>
#include <rk_mpi_mmz.h>
#include <rk_mpi_rgn.h>
#include <rk_mpi_sys.h>
#include <rk_mpi_tde.h>
#include <rk_mpi_venc.h>
#include <rk_mpi_vi.h>
#include <rk_mpi_vpss.h>
#include <rockiva.h>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <time.h>
#include <roi.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include "rknn_api.h"
#include "yolo_rknn_run.h"
#include "yolo_post_process.h"
#include "log.h"

#define LOG_TAG "YOLO"

#define YOLOV5_PATH	"/oem/usr/share//model/yolov5.rknn"
#define LABELS_PATH "/oem/usr/share//model/labels.txt"
#define ANCHORS_PATH "/oem/usr/share//model/anchors.txt"
#define VI_PIPE_ID 0
#define VI_CHN_ID 2
#define RGN_HANDLE_ID 8
#define ENABLE_YOLO_RGN 0

void *yolo_thread(void *arg);
static float sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

static unsigned long micros() {
    struct timespec time = {0, 0};
    
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1000000 + time.tv_nsec / 1000; /* microseconds */
}

static float get_interval() {
    static unsigned long last_time = 0;
    unsigned long now = micros();

    float result;

    if(last_time == 0)
    {
        result = 0;
    }else{
        result = now-last_time;
    }
    last_time = now;
    return result / 1000;
}

static float get_fps() {
    static unsigned long last_time = 0;
    unsigned long now = micros();

    float result;

    if(last_time == 0)
    {
        result = 0;
    }else{
        result = 1000000.0 / (now-last_time);
    }
    last_time = now;
    return result;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  char dims[128] = {0};
  for (int i = 0; i < attr->n_dims; ++i)
  {
    int idx = strlen(dims);
    sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
  }
  LOG_INFO("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, get_format_string(attr->fmt),
         get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


// 量化模型的npu输出结果为int8数据类型，后处理要按照int8数据类型处理
// 如下提供了int8排布的NC1HWC2转换成int8的nchw转换代码
static int NC1HWC2_int8_to_NCHW_int8(const int8_t *src, int8_t *dst, int *dims, int channel, int h, int w)
{
  int batch = dims[0];
  int C1 = dims[1];
  int C2 = dims[4];
  int hw_src = dims[2] * dims[3];
  int hw_dst = h * w;
  for (int i = 0; i < batch; i++)
  {
    src = src + i * C1 * hw_src * C2;
    dst = dst + i * channel * hw_dst;
    for (int c = 0; c < channel; ++c)
    {
      int plane = c / C2;
      const int8_t *src_c = plane * hw_src * C2 + src;
      int offset = c % C2;
      for (int cur_h = 0; cur_h < h; ++cur_h)
        for (int cur_w = 0; cur_w < w; ++cur_w)
        {
          int cur_hw = cur_h * w + cur_w;
          dst[c * hw_dst + cur_h * w + cur_w] = src_c[C2 * cur_hw + offset];
        }
    }
  }

  return 0;
}

static uint8_t yolo_init_flag = 0;
static uint8_t yolo_run_flag = 0;
static uint8_t nn_debug = 1;
static detect_result_t *results = NULL;
static rknn_context ctx = 0;
static rknn_input_output_num io_num;
static rknn_tensor_attr *input_attrs = NULL;
static rknn_tensor_attr *output_attrs_vector = NULL;
static rknn_tensor_mem *input_mems[1]={0};
static rknn_tensor_mem **output_mems = NULL;
static rga_buffer_t rga_src = {0};
static rga_buffer_t rga_dst = {0};
static int video0_width = 0;
static int video0_height = 0;
static int rknn_width = 0;
static int rknn_height = 0;

static uint64_t start_time = 0;
static uint64_t end_time = 0;
static uint32_t process_count = 0;
#define CACHE_RESULT_COUT_NUM 15
static uint8_t result_cache_index = 0;
static uint8_t result_count_cache[CACHE_RESULT_COUT_NUM] = {0}; 
extern uint64_t get_monotonic_timestamp_ms(void);
void update_canvas(detect_result_t *results, int count);
static pthread_mutex_t data_mutex = PTHREAD_MUTEX_INITIALIZER;
static sem_t data_sem;

int yolov5_post_process_func(void **output_mems_nchw, rknn_tensor_attr *output_attrs_vector, 
                    int output_cnt, int width, int height)
{
   
    int result_count = post_process(output_mems_nchw, width, height, output_attrs_vector, output_cnt, results);
    end_time = get_monotonic_timestamp_ms();
    process_count++;

    result_count_cache[result_cache_index] = result_count;
    result_cache_index = (result_cache_index + 1) % CACHE_RESULT_COUT_NUM;

#if ENABLE_YOLO_RGN
    update_canvas(results, result_count);
#endif    

    return result_count;
}

int check_yolo_detect_cache(void)
{
    for (int i = 0; i < CACHE_RESULT_COUT_NUM; i++)
    {
        if (result_count_cache[i] > 0)
        {
            return 1;
        }
    }
    return 0;
}

static int create_empty_canvas(RGN_CANVAS_INFO_S *canvas)
{
    memset(canvas, 0, sizeof(RGN_CANVAS_INFO_S));
    int ret = RK_MPI_RGN_GetCanvasInfo(RGN_HANDLE_ID, canvas);
    if (ret != RK_SUCCESS) {
        LOG_ERROR("RK_MPI_RGN_GetCanvasInfo failed with %#x!\n", ret);
        return ret;
    }
    memset((void*)canvas->u64VirAddr, 0x00, canvas->u32VirWidth * canvas->u32VirHeight * 2);
    return ret;
}

static void draw_hline(RGN_CANVAS_INFO_S *canvas, int x1, int x2, int y, uint16_t color)
{
    uint16_t *start = (uint16_t *)canvas->u64VirAddr + y*canvas->u32VirWidth;
    for(int x=x1; x<x2; x++)
    {
        *(start + x ) = color;
    }
}

static void draw_vline(RGN_CANVAS_INFO_S *canvas, int y1, int y2, int x, uint16_t color)
{
    uint16_t *start = (uint16_t *)canvas->u64VirAddr + x;
    for(int y=y1; y<y2; y++)
    {
        *(start + y*canvas->u32VirWidth ) = color;
    }
}

static int draw_cnt = 0;
void update_canvas(detect_result_t *results, int count)
{
    if (draw_cnt++ % 50 != 0)
    {
        return;
    }
    draw_cnt = 0;

    RK_S32 ret;
    RGN_CANVAS_INFO_S stCanvasInfo;
    ret = create_empty_canvas(&stCanvasInfo);
    if (ret != RK_SUCCESS) {
        LOG_ERROR("create_empty_canvas failed with %#x!\n", ret);
    }
    else
    {
        int x1, x2, y1, y2;
        for (int i=0; i<count; i++)
        {
            if (rknn_width != video0_width || rknn_height != video0_height)
            {
                x1 = results[i].left * video0_width / rknn_width;
                x2 = results[i].right * video0_width / rknn_width;
                y1 = results[i].top * video0_height / rknn_height;
                y2 = results[i].bottom * video0_height / rknn_height;
            }
            else
            {
                x1 = results[i].left;
                x2 = results[i].right;
                y1 = results[i].top;
                y2 = results[i].bottom;
            }
            
            draw_hline(&stCanvasInfo, x1, x2, y1, 0xFFFF);
            draw_hline(&stCanvasInfo, x1, x2, y2, 0xFFFF);
            draw_vline(&stCanvasInfo, y1, y2, x1, 0xFFFF);
            draw_vline(&stCanvasInfo, y1, y2, x2, 0xFFFF);
        }


        ret = RK_MPI_RGN_UpdateCanvas(RGN_HANDLE_ID);
        if (ret != RK_SUCCESS) {
            LOG_ERROR("RK_MPI_RGN_UpdateCanvas failed with %#x!\n", ret);
        }
    }
}


static RK_S32 rgn_init() 
{
    RK_S32 s32Ret = RK_SUCCESS;
	RGN_HANDLE RgnHandle = RGN_HANDLE_ID;
	RGN_ATTR_S stRgnAttr;
	RGN_CHN_ATTR_S stRgnChnAttr;

	int u32Width = video0_width;
	int u32Height = video0_height;
	int s32X = 0;
	int s32Y = 0;


	MPP_CHN_S stMppChn;
	stMppChn.enModId = RK_ID_VENC;
	stMppChn.s32DevId = 0;
	stMppChn.s32ChnId = 0;
    /****************************************
	step 1: create overlay regions
	****************************************/
    memset(&stRgnAttr, 0, sizeof(stRgnAttr));
	stRgnAttr.enType = OVERLAY_RGN;
	stRgnAttr.unAttr.stOverlay.enPixelFmt = (PIXEL_FORMAT_E)RK_FMT_BGRA5551;
	stRgnAttr.unAttr.stOverlay.stSize.u32Width = u32Width;
	stRgnAttr.unAttr.stOverlay.stSize.u32Height = u32Height;
	stRgnAttr.unAttr.stOverlay.u32ClutNum = 0;
    stRgnAttr.unAttr.stOverlay.u32CanvasNum = 1;

	s32Ret = RK_MPI_RGN_Create(RgnHandle, &stRgnAttr);
	if (RK_SUCCESS != s32Ret) {
		printf("RK_MPI_RGN_Create (%d) failed with %#x!", RgnHandle, s32Ret);
		RK_MPI_RGN_Destroy(RgnHandle);
		return RK_FAILURE;
	}
	printf("The handle: %d, create success!", RgnHandle);

	/*********************************************
	step 2: display overlay regions to groups
	*********************************************/
	memset(&stRgnChnAttr, 0, sizeof(stRgnChnAttr));
	stRgnChnAttr.bShow = RK_TRUE;
	stRgnChnAttr.enType = OVERLAY_RGN;
	stRgnChnAttr.unChnAttr.stOverlayChn.stPoint.s32X = s32X;
	stRgnChnAttr.unChnAttr.stOverlayChn.stPoint.s32Y = s32Y;
	stRgnChnAttr.unChnAttr.stOverlayChn.u32BgAlpha = 0;
	stRgnChnAttr.unChnAttr.stOverlayChn.u32FgAlpha = 255;
	stRgnChnAttr.unChnAttr.stOverlayChn.u32Layer = 0;
	stRgnChnAttr.unChnAttr.stOverlayChn.stQpInfo.bEnable = RK_FALSE;
	stRgnChnAttr.unChnAttr.stOverlayChn.stQpInfo.bForceIntra = RK_TRUE;
	stRgnChnAttr.unChnAttr.stOverlayChn.stQpInfo.bAbsQp = RK_FALSE;
	stRgnChnAttr.unChnAttr.stOverlayChn.stQpInfo.s32Qp = RK_FALSE;
	stRgnChnAttr.unChnAttr.stOverlayChn.u32ColorLUT[0] = 0x00;
	stRgnChnAttr.unChnAttr.stOverlayChn.u32ColorLUT[1] = 0xFFFFFF;
	stRgnChnAttr.unChnAttr.stOverlayChn.stInvertColor.bInvColEn = RK_FALSE;
	stRgnChnAttr.unChnAttr.stOverlayChn.stInvertColor.stInvColArea.u32Width = 16;
	stRgnChnAttr.unChnAttr.stOverlayChn.stInvertColor.stInvColArea.u32Height = 16;
	stRgnChnAttr.unChnAttr.stOverlayChn.stInvertColor.enChgMod = LESSTHAN_LUM_THRESH;
	stRgnChnAttr.unChnAttr.stOverlayChn.stInvertColor.u32LumThresh = 100;
	s32Ret = RK_MPI_RGN_AttachToChn(RgnHandle, &stMppChn, &stRgnChnAttr);
	if (RK_SUCCESS != s32Ret) {
		printf("RK_MPI_RGN_AttachToChn (%d) failed with %#x!\n", RgnHandle, s32Ret);
		return RK_FAILURE;
	}
	printf("Display region to chn success!\n"); 

    	/*********************************************
	step 4: use update canvas interface
	*********************************************/
	RGN_CANVAS_INFO_S stCanvasInfo;
	memset(&stCanvasInfo, 0, sizeof(RGN_CANVAS_INFO_S));

	s32Ret = RK_MPI_RGN_GetCanvasInfo(RgnHandle, &stCanvasInfo);
	if (s32Ret != RK_SUCCESS) {
		printf("RK_MPI_RGN_GetCanvasInfo failed with %#x!\n", s32Ret);
		return RK_FAILURE;
	}
    else
    {
        printf("RK_MPI_RGN_GetCanvasInfo success\n");
    }

	memset((void *)(stCanvasInfo.u64VirAddr), 0x00,
					stCanvasInfo.u32VirWidth * stCanvasInfo.u32VirHeight*2);
	s32Ret = RK_MPI_RGN_UpdateCanvas(RgnHandle);
	if (s32Ret != RK_SUCCESS) {
		printf("RK_MPI_RGN_UpdateCanvas failed with %#x!\n", s32Ret);
		return RK_FAILURE;
	}
    else
    printf("RK_MPI_RGN_UpdateCanvas success\n");
    
    return RK_SUCCESS;
}

int yolo_init(void)
{
    int ret;
    int vi_width = rk_param_get_int("video.2:width", 1280);
    int vi_height = rk_param_get_int("video.2:height", 720);
    video0_width = rk_param_get_int("video.0:width", 1280);
    video0_height = rk_param_get_int("video.0:width", 720);;
#if ENABLE_YOLO_RGN
    rgn_init();
#endif    

    if (sem_init(&data_sem, 0, 0) != 0) 
    {
        LOG_ERROR("sem_init");
        return -1;
    }

    if (post_process_init() < 0)
    {
        LOG_ERROR("post_process_init fail\n");
        return -1;
    }

    results = malloc(sizeof(detect_result_t)*OBJ_NUMB_MAX_SIZE);
    if (results == NULL)
    {
        LOG_ERROR("malloc results fail\n");
        return -1;
    }

    ret = rknn_init(&ctx, (void*)YOLOV5_PATH, 0, 0, NULL);
    if (ret != RKNN_SUCC)
    {
        LOG_ERROR("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    if (nn_debug)
    {
        // Get sdk and driver version
        rknn_sdk_version sdk_ver;
        ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
        if (ret != RKNN_SUCC)
        {
            LOG_ERROR("rknn_query fail! ret=%d\n", ret);
            goto ERROR;
        }
        LOG_INFO("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

        // Get custom string
        rknn_custom_string custom_string;
        ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
        if (ret != RKNN_SUCC)
        {
            LOG_ERROR("rknn_query fail! ret=%d\n", ret);
            goto ERROR;
        }
        LOG_INFO("custom string: %s\n", custom_string.string);
    }

    // Get Model Input Output Info
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        LOG_ERROR("rknn_query fail! ret=%d\n", ret);
        goto ERROR;
    }
    LOG_INFO("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    LOG_INFO("input tensors:\n");
    input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            LOG_ERROR("rknn_init error! ret=%d\n", ret);
            goto ERROR;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    LOG_INFO("output tensors:\n");
    output_attrs_vector = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memset(output_attrs_vector, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attrs_vector[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_attrs_vector[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            LOG_ERROR("rknn_query fail! ret=%d\n", ret);
            goto ERROR;
        }
        dump_tensor_attr(&output_attrs_vector[i]);
    }

    // Create input tensor memory
    // default input type is int8 (normalize and quantize need compute in outside)
    // if set uint8, will fuse normalize and quantize to npu
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    // default fmt is NHWC, npu only support NHWC in zero copy mode
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

     // check model input
    int width = input_attrs[0].dims[2];
    int height = input_attrs[0].dims[1];
    int stride = input_attrs[0].w_stride;
    rknn_width = width;
    rknn_height = height;

    if (width != stride)
    {
        LOG_ERROR("width != stride\n");
        goto ERROR;
    }

    // Create output tensor memory
    output_mems = (rknn_tensor_mem **)malloc(io_num.n_output * sizeof(rknn_tensor_mem *));
    memset(output_mems, 0, io_num.n_output * sizeof(rknn_tensor_mem*));
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        output_mems[i] = rknn_create_mem(ctx, output_attrs_vector[i].size_with_stride);
    }

    // Set input tensor memory
    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret != RKNN_SUCC)
    {
        LOG_ERROR("rknn_set_io_mem fail! ret=%d\n", ret);
        goto ERROR;
    }

    // Set output tensor memory
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        // set output memory and attribute
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs_vector[i]);
        if (ret != RKNN_SUCC)
        {
            LOG_ERROR("rknn_set_io_mem fail! ret=%d\n", ret);
            goto ERROR;
        }
    }

    // set rga
    rga_src.width = vi_width;
    rga_src.height = vi_height;
    rga_src.wstride = vi_width;
    rga_src.hstride = vi_height;
    rga_src.format = RK_FORMAT_YCbCr_420_SP;

    rga_dst.width = width;
    rga_dst.height = height;
    rga_dst.wstride = width;
    rga_dst.hstride = height;
    rga_dst.format = RK_FORMAT_RGB_888;
    rga_dst.phy_addr = (void*)input_mems[0]->phys_addr;

    yolo_init_flag = 1;
    start_time = get_monotonic_timestamp_ms();

    pthread_t pid;
    pthread_create(&pid, NULL, yolo_thread, NULL);

    return 0;

ERROR:
    rknn_destroy(ctx);
    free(input_attrs);
    free(output_attrs_vector);
    // Destroy rknn memory
    rknn_destroy_mem(ctx, input_mems[0]);
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        rknn_destroy_mem(ctx, output_mems[i]);
    }
    free(output_mems);
    free(results);
    input_attrs = NULL;
    output_attrs_vector = NULL;
    input_mems[0] = NULL;
    output_mems = NULL;
    results = NULL;

    return -1;
}

int yolo_deinit(void)
{
    sem_destroy(&data_sem);
    yolo_init_flag = 0;
    yolo_run_flag = 0;
    rknn_destroy(ctx);
    free(input_attrs);
    free(output_attrs_vector);
    rknn_destroy_mem(ctx, input_mems[0]);
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        rknn_destroy_mem(ctx, output_mems[i]);
    }
    free(output_mems);
    input_attrs = NULL;
    output_attrs_vector = NULL;
    input_mems[0] = NULL;
    output_mems = NULL;
    
}

int yolo_run(VIDEO_FRAME_INFO_S *pstViFrameInfo)
{
    int ret, i;
    int ret_val = -1;
    rknn_tensor_attr *orig_output_attrs = NULL;
    void **output_mems_nchw = NULL;
    if (!yolo_init_flag)
        return -1;

    int width = input_attrs[0].dims[2];
    int height = input_attrs[0].dims[1];

    #if 0 //cal fps
        static int run_count = 0;
        static uint64_t last_time = 0;
        uint64_t now = get_monotonic_timestamp_ms();
        if (last_time == 0)
        {
            last_time = now;
        }
        if (run_count++ > 50)
        {
            LOG_INFO("fps: %f\n", run_count*1000.0/(now-last_time));
            last_time = now;
            run_count = 0;
        }
    #endif

    #if 0
    rga_src.phy_addr = (void*)RK_MPI_MB_Handle2PhysAddr(pstViFrameInfo->stVFrame.pMbBlk);
    ret = imcvtcolor(rga_src, rga_dst, rga_src.format, rga_dst.format);
    if (ret < 0)
    {
        LOG_ERROR("imcvtcolor error %d\n", ret);
        return -1;
    }
    #endif

    ret = rknn_run(ctx, NULL);
    if (ret < 0)
    {
        LOG_ERROR("rknn run error %d\n", ret);
        return -1;
    }
    // printf("%.2fms, rknn_run\n", get_interval());

    orig_output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memset(orig_output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));

    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        orig_output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(orig_output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            LOG_ERROR("rknn_query fail! ret=%d\n", ret);
            goto END;
        }
        // dump_tensor_attr(&orig_output_attrs[i]);
    }
    // malloc output nchw data
    output_mems_nchw = (void **)malloc(io_num.n_output * sizeof(void *));
    for (i = 0; i < io_num.n_output; ++i)
    {
        int size = orig_output_attrs[i].size_with_stride;
        output_mems_nchw[i] = malloc(size);
    }

    for (i = 0; i < io_num.n_output; i++)
    {
        int channel = orig_output_attrs[i].dims[1];
        int h = orig_output_attrs[i].n_dims > 2 ? orig_output_attrs[i].dims[2] : 1;
        int w = orig_output_attrs[i].n_dims > 3 ? orig_output_attrs[i].dims[3] : 1;
        int hw = h * w;
        NC1HWC2_int8_to_NCHW_int8((int8_t *)output_mems[i]->virt_addr, (int8_t *)output_mems_nchw[i], (int *)output_attrs_vector[i].dims,
                                channel, h, w);
    }

    yolov5_post_process_func(output_mems_nchw, output_attrs_vector, io_num.n_output, width, height);

END:
    free(orig_output_attrs);
    for (i = 0; i < io_num.n_output; ++i)
    {
        free(output_mems_nchw[i]);
    }
    free(output_mems_nchw);

    ret_val = 0;
    return ret_val;    
}

int set_yolo_img_data(VIDEO_FRAME_INFO_S *pstViFrameInfo)
{
    int ret;
    ret = pthread_mutex_trylock(&data_mutex);
    if (ret == 0)
    {
        rga_src.phy_addr = (void*)RK_MPI_MB_Handle2PhysAddr(pstViFrameInfo->stVFrame.pMbBlk);
        ret = imcvtcolor(rga_src, rga_dst, rga_src.format, rga_dst.format);
        if (ret >= 0)
        {
             sem_post(&data_sem);
        }
        pthread_mutex_unlock(&data_mutex);
    }
   
   return ret;
}

static MB_BLK mb = NULL;
static RK_U8 *vaddr = NULL;
static RK_U64 paddr = 0;

void set_yolo_img_data_frome_viCache_init(void)
{
    int ret = 0;
    int vi_width = rk_param_get_int("video.2:width", 1280);
    int vi_height = rk_param_get_int("video.2:height", 720);
	int len = vi_width*vi_height*3/2; //RK_FORMAT_YCbCr_420_SP

    ret = RK_MPI_MMZ_Alloc(&mb, len, RK_MMZ_ALLOC_TYPE_CMA);
    if (ret < 0) 
    {
        mb = NULL;
        vaddr = NULL;
        paddr = 0;
        printf("alloc mmz fail");
        return ret;
    }

    vaddr = (RK_U8 *)RK_MPI_MMZ_Handle2VirAddr(mb);
    paddr = RK_MPI_MMZ_Handle2PhysAddr(mb);
}

void set_yolo_img_data_frome_viCache_deinit(void)
{
    RK_MPI_MMZ_Free(mb);
    mb = NULL;
    vaddr = NULL;
    paddr = 0;
}

int set_yolo_img_data_frome_viCache(void *data, int len)
{
    int ret = 0;

    if (!vaddr)
    return -1;

    memcpy(vaddr, data, len);
    pthread_mutex_lock(&data_mutex);
    rga_src.phy_addr = (void*)paddr;
    ret = imcvtcolor(rga_src, rga_dst, rga_src.format, rga_dst.format);
    if (ret >= 0)
    {
            sem_post(&data_sem);
    }
    pthread_mutex_unlock(&data_mutex);
}

void *yolo_thread(void *arg)
{
    int ret;
    VIDEO_FRAME_INFO_S stViFrameInfo;
    yolo_run_flag = 1;
    start_time = get_monotonic_timestamp_ms();
    while (yolo_run_flag)
    {
        sem_wait(&data_sem);
        pthread_mutex_trylock(&data_mutex);
        yolo_run(NULL);
        pthread_mutex_unlock(&data_mutex);
    }
    return NULL;
}
