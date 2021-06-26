object FormConditions: TFormConditions
  Left = 0
  Top = 0
  Caption = 'Conditions'
  ClientHeight = 299
  ClientWidth = 635
  Color = clMoneyGreen
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OldCreateOrder = False
  PixelsPerInch = 96
  TextHeight = 13
  object Label1: TLabel
    Left = 24
    Top = 24
    Width = 101
    Height = 13
    Caption = 'ambient temperature'
  end
  object Label2: TLabel
    Left = 232
    Top = 24
    Width = 7
    Height = 13
    Caption = 'C'
  end
  object Label3: TLabel
    Left = 24
    Top = 56
    Width = 86
    Height = 13
    Caption = 'power dissipation '
  end
  object Label4: TLabel
    Left = 232
    Top = 56
    Width = 10
    Height = 13
    Caption = 'W'
  end
  object Edit1: TEdit
    Left = 144
    Top = 21
    Width = 73
    Height = 21
    TabOrder = 0
    Text = '22'
  end
  object Edit2: TEdit
    Left = 144
    Top = 48
    Width = 73
    Height = 21
    TabOrder = 1
    Text = '1'
  end
  object Button1: TButton
    Left = 144
    Top = 184
    Width = 75
    Height = 25
    Caption = 'Apply'
    TabOrder = 2
    OnClick = Button1Click
  end
end
