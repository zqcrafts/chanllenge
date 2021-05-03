rm widerface_txt.zip
zip -r widerface_txt.zip widerface_txt
HDFS_PREFIX=hdfs://haruna/home/byte_labcv_default/user/chenzehui.123/code/workspace
hadoop fs -put -f widerface_txt.zip ${HDFS_PREFIX}/