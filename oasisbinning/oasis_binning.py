#!/usr/bin/env python3
"""
终极修正版 OASIS文件合并脚本 - 解决所有合并问题及拼接缝问题
此脚本可以自动将一个或多个文件夹中的所有OASIS文件合并成大文件，
然后将各个文件夹的合并结果再次合并成一个最终的大OASIS文件。
支持无缝拼接。
"""

import os
import glob
from pathlib import Path
import pya  # KLayout Python API


def get_oasis_files(folder_path):
    """
    获取文件夹中所有的OASIS文件

    Args:
        folder_path (str): 文件夹路径

    Returns:
        list: OASIS文件路径列表
    """
    oasis_extensions = ['*.oas', '*.oasi', '*.oa']
    oasis_files = []

    for ext in oasis_extensions:
        oasis_files.extend(glob.glob(os.path.join(folder_path, ext), recursive=True))
        oasis_files.extend(glob.glob(os.path.join(folder_path, ext.upper()), recursive=True))

    return sorted(oasis_files)


def copy_cell_content_with_transform(source_cell, target_cell, target_layout, source_layout, trans=pya.Trans()):
    """
    将源单元格内容复制到目标单元格，包含所有图层的内容

    Args:
        source_cell: 源单元格
        target_cell: 目标单元格
        target_layout: 目标布局
        source_layout: 源布局
        trans: 变换矩阵
    """
    # 遍历源布局的所有图层
    for layer_idx in range(source_layout.layers()):
        if not source_layout.is_valid_layer(layer_idx):
            continue

        layer_info = source_layout.get_info(layer_idx)

        # 在目标布局中获取或创建对应的图层
        try:
            target_layer = target_layout.layer(layer_info)
        except:
            target_layer = target_layout.layer(layer_info.layer, layer_info.datatype)

        # 获取源单元格在该图层上的形状
        source_shapes = source_cell.shapes(layer_idx)
        target_shapes = target_cell.shapes(target_layer)

        # 复制所有形状
        for shape in source_shapes.each():
            if shape.is_box():
                target_shapes.insert(shape.box.transformed(trans))
            elif shape.is_polygon():
                target_shapes.insert(shape.polygon.transformed(trans))
            elif shape.is_path():
                target_shapes.insert(shape.path.transformed(trans))
            elif shape.is_text():
                target_shapes.insert(shape.text.transformed(trans))
            elif shape.is_edge():
                target_shapes.insert(shape.edge.transformed(trans))
            elif shape.is_edge_pairs():
                target_shapes.insert(shape.edge_pairs.transformed(trans))
            elif shape.is_instance():
                inst = shape.instance
                new_trans = trans * inst.trans
                ref_cell_name = inst.cell.name

                # 检查目标布局中是否存在引用的单元格
                ref_target_cell = target_layout.cell(ref_cell_name)
                if ref_target_cell is None:
                    # 如果不存在，递归复制整个引用的单元格
                    ref_source_cell = inst.cell
                    ref_target_cell = target_layout.create_cell(ref_cell_name)
                    copy_cell_content_with_transform(ref_source_cell, ref_target_cell,
                                                     target_layout, source_layout, pya.Trans())

                target_shapes.insert(ref_target_cell, new_trans)
            else:
                # 对于其他类型的形状，也进行变换后插入
                target_shapes.insert(shape.transformed(trans))


def merge_layouts_simple(layout_paths, output_path):
    """
    简单的布局合并函数，将多个布局文件合并成一个
    完全重写以避免循环依赖问题，并实现无缝拼接

    Args:
        layout_paths: 布局文件路径列表
        output_path: 输出路径
    """
    # 创建目标布局
    target_layout = pya.Layout()

    # 设置默认DBU（数据库单位）
    target_layout.dbu = 0.001  # 1nm

    # 用于跟踪已添加的单元格名称，避免重名
    added_cells = {}
    cell_rename_map = {}  # 用于记录单元格重命名映射

    for file_idx, layout_path in enumerate(layout_paths):
        print(f"  正在加载: {os.path.basename(layout_path)}")

        # 临时加载源布局
        source_layout = pya.Layout()
        source_layout.read(layout_path)

        # 设置目标布局的DBU与第一个源布局一致（或者保持默认）
        if file_idx == 0:
            target_layout.dbu = source_layout.dbu

        # 复制所有单元格，避免重名
        for cell_index in range(source_layout.cells()):
            source_cell = source_layout.cell(cell_index)
            if source_cell:
                original_name = source_cell.name

                # 生成唯一的新名字，避免冲突
                new_name = original_name
                counter = 1
                while new_name in added_cells:
                    # 如果单元格名已存在，添加文件索引前缀
                    new_name = f"{file_idx:03d}_{original_name}_{counter}"
                    counter += 1

                # 创建目标单元格
                target_cell = target_layout.create_cell(new_name)

                # 记录重命名映射
                cell_rename_map[(file_idx, original_name)] = new_name
                added_cells[new_name] = True

                # 复制单元格内容
                copy_cell_content_with_transform(source_cell, target_cell,
                                                 target_layout, source_layout, pya.Trans())

        # 删除临时布局以释放内存
        del source_layout

    # 创建一个统一的顶层单元格来包含所有内容
    top_cell_name = "TOP_MERGED_LAYOUT"
    top_cell = target_layout.create_cell(top_cell_name)

    # 计算每个输入文件的总宽度，用于无缝拼接
    file_widths = []
    for file_idx, layout_path in enumerate(layout_paths):
        temp_layout = pya.Layout()
        temp_layout.read(layout_path)
        total_width = 0
        for cell_index in range(temp_layout.cells()):
            cell = temp_layout.cell(cell_index)
            if cell and cell.parent_cells() == 0:
                bbox = cell.bbox()
                if not bbox.empty():
                    total_width = max(total_width, bbox.width())
        file_widths.append(total_width)
        del temp_layout

    # 水平无缝拼接：将每个文件的顶层单元格按顺序平移
    x_offset = 0
    for file_idx, layout_path in enumerate(layout_paths):
        temp_layout = pya.Layout()
        temp_layout.read(layout_path)

        for cell_index in range(temp_layout.cells()):
            cell = temp_layout.cell(cell_index)
            if cell and cell.parent_cells() == 0:
                original_name = cell.name
                target_name = cell_rename_map.get((file_idx, original_name), original_name)
                target_cell = target_layout.cell(target_name)
                if target_cell:
                    # 使用累积的x偏移量进行平移，实现无缝拼接
                    trans = pya.Trans(x_offset, 0)
                    top_cell.insert(pya.CellInstArray(target_cell.cell_index(), trans))

        # 更新下一个文件的起始x位置
        if file_widths[file_idx] > 0:
            x_offset += file_widths[file_idx]

        del temp_layout

    # 保存结果
    print(f"  保存合并结果到: {output_path}")
    target_layout.write(output_path)

    # 清理
    del target_layout


def merge_folder_oasis_files(folder_path, output_path):
    """
    合并单个文件夹中的所有OASIS文件

    Args:
        folder_path (str): 输入文件夹路径
        output_path (str): 输出OASIS文件路径
    """
    print(f"正在处理文件夹: {folder_path}")

    # 获取文件夹中的所有OASIS文件
    oasis_files = get_oasis_files(folder_path)

    if not oasis_files:
        print(f"警告: 文件夹 {folder_path} 中未找到OASIS文件")
        return False

    print(f"找到 {len(oasis_files)} 个OASIS文件")

    # 使用简单合并函数
    merge_layouts_simple(oasis_files, output_path)

    print(f"文件夹 {folder_path} 合并完成")
    return True


def merge_specific_folders(folder_paths, output_base_path="./merged_output"):
    """
    处理指定的文件夹列表

    Args:
        folder_paths (list): 文件夹路径列表
        output_base_path (str): 输出基础路径
    """
    # 创建输出目录
    os.makedirs(output_base_path, exist_ok=True)

    # 第一步：合并每个文件夹中的OASIS文件
    temp_merged_files = []

    for i, folder_path in enumerate(folder_paths):
        if not os.path.isdir(folder_path):
            print(f"错误: 文件夹不存在 {folder_path}")
            continue

        temp_output_file = os.path.join(output_base_path, f"merged_folder_{i + 1}_{os.path.basename(folder_path)}.oas")

        success = merge_folder_oasis_files(folder_path, temp_output_file)
        if success:
            temp_merged_files.append(temp_output_file)

    # 第二步：将所有临时合并文件再次合并成最终文件
    if not temp_merged_files:
        print("错误: 没有成功合并任何文件夹")
        return

    print("\n开始最终合并...")
    print(f"将 {len(temp_merged_files)} 个中间文件合并为最终文件")

    # 合并所有临时文件
    final_output_path = os.path.join(output_base_path, "final_merged_all.oas")
    merge_layouts_simple(temp_merged_files, final_output_path)

    print(f"\n所有处理完成!")
    print(f"中间文件保存在: {output_base_path}")
    print(f"最终合并文件: {final_output_path}")


if __name__ == "__main__":
    print("OASIS文件合并工具 - 终极修正版（无缝拼接）")
    print("=" * 50)

    # 指定要处理的文件夹路径
    folders_to_process = [
        "/your/path/folder1",
        "/your/path/folder2",
        "/your/path/folder3"
        # 添加更多文件夹路径
    ]

    # 运行合并
    merge_specific_folders(folders_to_process, "./oasis_merge_output")