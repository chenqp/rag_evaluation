import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from logger_config import logger
'''
使用评估结果evaluation_results.xlsx，统计指标分布的直方图
'''
def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成评估指标分布直方图')
    parser.add_argument('--file', type=str, required=True, 
                        help='Excel文件路径 (必需)')
    return parser.parse_args()

def setup_output_directory(file_name):
    """创建输出目录"""
    output_dir = Path(f"{file_name}_metrcs_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    return output_dir

def get_metric_config():
    """获取指标配置"""
    columns_to_analyze = [
        'llm_context_precision_with_reference',
        'context_recall',
        'nv_context_relevance',
        'faithfulness',
        'nv_accuracy'
    ]
    
    short_names = {
        'llm_context_precision_with_reference': 'Context Precision',
        'context_recall': 'Context Recall',
        'nv_context_relevance': 'Context Relevance',
        'faithfulness': 'Faithfulness',
        'nv_accuracy': 'Accuracy'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    return columns_to_analyze, short_names, colors

def create_combined_plot(df, sheet_name, columns, short_names, colors, output_dir, file_name):
    """为单个工作表创建综合图表并保存"""
    logger.info(f"处理工作表: {sheet_name}")
    
    # 创建大图（5个子图垂直排列）
    fig, axes = plt.subplots(5, 1, figsize=(10, 15))
    fig.suptitle(f'Evaluation Metrics Distribution - {sheet_name}', fontsize=16, y=0.95)
    
    # 设置直方图分箱
    bins = np.arange(0, 1.1, 0.1)
    
    # 为每个指标创建直方图子图
    for i, col in enumerate(columns):
        ax = axes[i]
        
        # 绘制直方图
        ax.hist(df[col], bins=bins, color=colors[i], 
                edgecolor='black', alpha=0.8, rwidth=0.9)
        
        # 添加标题和标签
        ax.set_title(f'{short_names[col]}', fontsize=12)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_xlim(0, 1)
        
        # 设置Y轴为整数
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 添加网格线
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在柱子上方添加计数标签
        counts, _ = np.histogram(df[col], bins=bins)
        max_count = max(counts) if len(counts) > 0 and max(counts) > 0 else 1
        ax.set_ylim(0, max_count * 1.15)  # 为计数标签留出空间
        
        for j in range(len(counts)):
            if counts[j] > 0:
                ax.text(bins[j] + 0.05, counts[j] + max_count*0.05, str(counts[j]), 
                         fontsize=9, ha='center', fontweight='bold')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    
    # 保存图像
    plot_filename = f"{file_name}_{sheet_name}_metrics.png"
    plt.savefig(output_dir / plot_filename, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"已保存: {plot_filename}")

def process_excel_file(file_path):
    """处理Excel文件"""
    # 验证文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    file_name = file_path.stem  # 获取不带扩展名的文件名
    logger.info(f"正在处理文件: {file_path}")
    
    # 读取所有sheet名称
    all_sheets = pd.ExcelFile(file_path).sheet_names
    
    # 获取指标配置
    columns_to_analyze, short_names, colors = get_metric_config()
    
    # 设置输出目录
    output_dir = setup_output_directory(file_name)
    
    # 处理每个sheet
    for sheet_name in all_sheets:
        # 读取当前sheet数据
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 创建并保存综合图表
        create_combined_plot(
            df, sheet_name, 
            columns_to_analyze, short_names, colors, 
            output_dir, file_name
        )

def main():
    """主函数"""
    args = parse_arguments()
    file_path = Path(args.file)
    
    try:
        process_excel_file(file_path)
        logger.info(f"\n处理完成! 所有图表已保存到 {file_path.stem}_metrcs_plots 目录中")
    except Exception as e:
        logger.error(f"\n处理过程中出错: {e}")

if __name__ == "__main__":
    main()