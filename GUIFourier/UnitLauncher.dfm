object FormLauncher: TFormLauncher
  Left = 0
  Top = 0
  Caption = 'Launcher'
  ClientHeight = 186
  ClientWidth = 418
  Color = clMoneyGreen
  CustomTitleBar.CaptionAlignment = taCenter
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OldCreateOrder = False
  PixelsPerInch = 96
  TextHeight = 13
  object Label1: TLabel
    Left = 16
    Top = 158
    Width = 31
    Height = 13
    Caption = 'GPU id'
  end
  object RadioGroup1: TRadioGroup
    Left = 8
    Top = 8
    Width = 185
    Height = 139
    Caption = 'Precision'
    ItemIndex = 0
    Items.Strings = (
      'CPU float'
      'CPU double'
      'GPU NVIDIA float')
    TabOrder = 0
  end
  object ButtonApply: TButton
    Left = 118
    Top = 153
    Width = 75
    Height = 25
    Caption = 'Apply'
    TabOrder = 1
    OnClick = ButtonApplyClick
  end
  object ComboBoxGPUid: TComboBox
    Left = 53
    Top = 157
    Width = 36
    Height = 21
    ItemIndex = 0
    TabOrder = 2
    Text = '0'
    Items.Strings = (
      '0'
      '1'
      '2'
      '3'
      '4'
      '5')
  end
end
